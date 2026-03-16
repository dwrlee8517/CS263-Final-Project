import csv
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, list_repo_files
from openai import OpenAI
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm import tqdm

load_dotenv()


MODEL_NAMES = [
    "gpt-5-nano",
    "models/gemini-2.5-flash-lite",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
]

FINAL_CONFIG = {
    "sample_size": None,  # None means full split
    "seed": 42,
    "split": "test",
    "temperature": 0.0,
    "max_tokens": 16,
    "retry_on_invalid": 1,
    "reasoning_effort": "low",
}

EVAL_LABEL_TO_ID = {"true": 0, "false": 1, "mixture": 2}
INTEL_TO_EVAL_LABEL = {
    "true": "true",
    "false": "false",
    "mostly_true": "mixture",
    "partially_true": "mixture",
}
PUBHEALTH_TO_EVAL_LABEL = {
    0: "true",
    1: "false",
    2: None,  # unproven; excluded from 3-class eval
    3: "mixture",
}


def load_project_paths(main_dir: str = ".") -> Dict[str, str]:
    base_model_path = os.path.join(main_dir, "model", "bertweet_base_local")
    checkpoint_dir = os.path.join(main_dir, "model", "checkpoints")
    dataset_dir = os.path.join(main_dir, "data")
    os.makedirs(base_model_path, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(dataset_dir, exist_ok=True)
    return {
        "main_dir": main_dir,
        "base_model_path": base_model_path,
        "checkpoint_dir": checkpoint_dir,
        "dataset_dir": dataset_dir,
    }


def download_pubhealth_splits(dataset_dir: str, revision: str = "refs/convert/parquet") -> Tuple[DatasetDict, DatasetDict]:
    os.makedirs(dataset_dir, exist_ok=True)
    repo_files = list_repo_files("bigbio/pubhealth", repo_type="dataset", revision=revision)

    def download_split_files(subset_name: str):
        parquet_files = [f for f in repo_files if f.startswith(subset_name) and f.endswith(".parquet")]
        if not parquet_files:
            raise RuntimeError(f"No parquet files found for subset {subset_name} at {revision}")

        split_files = {"train": [], "validation": [], "test": []}
        for f in parquet_files:
            if "/train/" in f:
                split = "train"
            elif "/validation/" in f:
                split = "validation"
            elif "/test/" in f:
                split = "test"
            else:
                split = "train"

            local_path = os.path.join(dataset_dir, f)
            if os.path.exists(local_path):
                split_files[split].append(local_path)
            else:
                split_files[split].append(
                    hf_hub_download(
                        repo_id="bigbio/pubhealth",
                        repo_type="dataset",
                        filename=f,
                        revision=revision,
                        local_dir=dataset_dir,
                        local_dir_use_symlinks=False,
                    )
                )

        return {k: v for k, v in split_files.items() if v}

    source_files = download_split_files("pubhealth_source")
    pair_files = download_split_files("pubhealth_bigbio_pairs")
    source_ds = load_dataset("parquet", data_files=source_files)
    pair_ds = load_dataset("parquet", data_files=pair_files)
    return source_ds, pair_ds


def load_intel_dataset(dataset_dir: str) -> DatasetDict:
    intel_ds = load_dataset("Intel/misinformation-guard")
    int_to_text_map = {
        0: "false",
        1: "mostly_true",
        2: "partially_true",
        3: "true",
    }
    intel_ds = intel_ds.map(lambda x: {"label_text": int_to_text_map[x["label"]]})
    intel_save_path = os.path.join(dataset_dir, "intel_misinformation_guard")
    os.makedirs(intel_save_path, exist_ok=True)
    intel_ds.save_to_disk(intel_save_path)
    return intel_ds


def build_zero_shot_messages(claim_text: str) -> List[Dict]:
    return [
        {"role": "system", "content": "You are a strict fact-checking assistant that only responds with labels."},
        {
            "role": "user",
            "content": (
                "Classify the following claim into exactly one of: true, false, mixture. "
                "Do not add any explanation, just output the label.\n\n"
                f"Claim: {claim_text}\n"
                "Label:"
            ),
        },
    ]


def build_few_shot_messages(claim_text: str, few_shot_examples: List[Dict]) -> List[Dict]:
    if not few_shot_examples:
        raise ValueError("few_shot_examples is empty. Add examples in evaluate_config.yaml.")

    messages: List[Dict] = [
        {
            "role": "system",
            "content": (
                """
                You are an expert medical/health information fact-checker. Your task is to classify claims based on their factual accuracy into one of three categories: true, false, or mixture.

                ### Label Definitions:
                - **true**: The claim is entirely accurate, supported by evidence, and contains no significant omissions or misleading context.
                - **false**: The claim is entirely inaccurate, fabricated, or fundamentally contradicted by established facts.
                - **mixture**: The claim contains elements of truth but is problematic. This includes:
                    * Claims that are partially true and partially false.
                    * Claims that are "mostly true" but contain minor inaccuracies.
                    * Claims that are "partially true" but lack critical context or use a misleading framing.

                ### Constraints:
                - Output exactly one word: "true", "false", or "mixture".
                - Do not provide explanations or punctuation.
                """
            ),
        }
    ]

    valid_labels = {"true", "false", "mixture"}
    for idx, ex in enumerate(few_shot_examples):
        if not isinstance(ex, dict):
            raise ValueError(f"few_shot_examples[{idx}] must be a mapping with 'claim' and 'label'.")
        claim = str(ex.get("claim", "")).strip()
        label = str(ex.get("label", "")).strip().lower()
        if not claim:
            raise ValueError(f"few_shot_examples[{idx}].claim is missing.")
        if label not in valid_labels:
            raise ValueError(
                f"few_shot_examples[{idx}].label must be one of {sorted(valid_labels)}, got '{label}'."
            )
        messages.append({"role": "user", "content": f"Claim: {claim}\nLabel:"})
        messages.append({"role": "assistant", "content": label})

    messages.append({"role": "user", "content": f"Claim: {claim_text}\nLabel:"})
    return messages


def build_prompt_messages(claim_text: str, config: Dict) -> List[Dict]:
    prompt_mode = str(config.get("prompt_mode", "zero_shot")).strip().lower()
    if prompt_mode == "few_shot":
        return build_few_shot_messages(claim_text, config.get("few_shot_examples", []))
    return build_zero_shot_messages(claim_text)


def parse_label(raw_text: str) -> str | None:
    if not raw_text:
        return None
    cleaned = raw_text.strip().lower()
    cleaned = re.sub(r"[^a-z_ ]", " ", cleaned)
    cleaned = " ".join(cleaned.split())
    if cleaned in EVAL_LABEL_TO_ID:
        return cleaned
    if "partially true" in cleaned or "mostly true" in cleaned or "mixed" in cleaned:
        return "mixture"
    tokens = cleaned.split()
    if len(tokens) > 3:
        last_token = tokens[-1]
        if last_token in EVAL_LABEL_TO_ID:
            return last_token
        if last_token in {"partially_true", "mostly_true", "mixed"}:
            return "mixture"
        if len(tokens) >= 2:
            last_bigram = " ".join(tokens[-2:])
            if last_bigram in {"partially true", "mostly true"}:
                return "mixture"
    for token in tokens:
        if token in EVAL_LABEL_TO_ID:
            return token
        if token in {"partially_true", "mostly_true"}:
            return "mixture"
    return None


def _generate_openai_response(messages: List[Dict], model: str, **gen_params) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    reasoning_value = gen_params.get("reasoning_effort") or gen_params.get("reasoning")
    reasoning_param = {"effort": reasoning_value} if reasoning_value else None
    request_params = dict(gen_params)
    for k in ("model", "reasoning", "reasoning_effort", "temperature", "top_p", "frequency_penalty", "presence_penalty"):
        request_params.pop(k, None)
    max_tokens = request_params.pop("max_tokens", None)
    # if max_tokens is not None:
    #     request_params["max_output_tokens"] = max_tokens
    response = client.responses.create(model=model, reasoning=reasoning_param, input=messages, **request_params)
    return response.output_text or ""


def _generate_deepinfra_response(messages: List[Dict], model: str, **gen_params) -> str:
    api_key = os.getenv("DEEPINFRA_TOKEN")
    client = OpenAI(api_key=api_key, base_url="https://api.deepinfra.com/v1/openai")
    if "gemini" in model.lower():
        completion = client.chat.completions.create(model=model, messages=messages, stream=False)
    else:
        completion = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=gen_params.get("temperature", 0.0),
            max_tokens=gen_params.get("max_tokens", 16),
            stream=False,
        )
    return completion.choices[0].message.content or ""


def _generate_featherless_response(messages: List[Dict], model: str, **gen_params) -> str:
    api_key = os.getenv("FEATHERLESS_API_KEY")
    if not api_key:
        raise ValueError("FEATHERLESS_API_KEY is missing. Set it in your environment.")

    client = OpenAI(api_key=api_key, base_url="https://api.featherless.ai/v1")
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=gen_params.get("temperature", 0.0),
        max_tokens=gen_params.get("max_tokens", 16),
        stream=False,
    )
    return completion.choices[0].message.content or ""


def _generate_google_gemini_response(messages: List[Dict], model: str, **gen_params) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    client = OpenAI(api_key=api_key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    kwargs = {"model": model, "messages": messages, "stream": False}
    reasoning_effort = gen_params.get("reasoning_effort") or gen_params.get("reasoning")
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    completion = client.chat.completions.create(**kwargs)
    return completion.choices[0].message.content or ""


def generate_model_response(messages: List[Dict], model: str, **gen_params) -> str:
    if "gpt" in model.lower():
        return _generate_openai_response(messages, model, **gen_params)
    if "gemini" in model.lower():
        return _generate_google_gemini_response(messages, model, **gen_params)
    try:
        return _generate_featherless_response(messages, model, **gen_params)
    except Exception as first_err:
        fallback_msg = str(first_err)
        print(f"Featherless request failed, falling back to DeepInfra: {fallback_msg}")
    return _generate_deepinfra_response(messages, model, **gen_params)


def safe_model_label(messages: List[Dict], model: str, retries: int = 1, **gen_params) -> Tuple[str | None, str]:
    last_raw = ""
    for _ in range(retries + 1):
        try:
            raw = generate_model_response(messages, model=model, **gen_params)
        except Exception as e:
            last_raw = f"ERROR[{model}]: {e}"
            continue
        last_raw = raw or ""
        parsed = parse_label(last_raw)
        if parsed is not None:
            return parsed, last_raw
    return None, last_raw


def sample_or_full_split(split_ds: Dataset, sample_size: int | None, seed: int) -> Dataset:
    if sample_size is None:
        return split_ds
    n = len(split_ds)
    if sample_size >= n:
        return split_ds
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(n), sample_size))
    return split_ds.select(indices)


def compute_stats(rows: List[Dict], model: str, dataset_name: str, split: str, elapsed_sec: float) -> Dict:
    valid_rows = [r for r in rows if r["pred_label"] is not None]
    coverage = (len(valid_rows) / len(rows)) if rows else 0.0
    stats = {
        "model": model,
        "dataset": dataset_name,
        "split": split,
        "elapsed_sec": elapsed_sec,
        "sample_count": len(rows),
        "valid_count": len(valid_rows),
        "coverage": coverage,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    if not valid_rows:
        stats["accuracy"] = None
        stats["macro_f1"] = None
        stats["confusion_matrix"] = []
        stats["classification_report"] = {}
        return stats

    y_true = np.array([EVAL_LABEL_TO_ID[r["gold_label"]] for r in valid_rows])
    y_pred = np.array([EVAL_LABEL_TO_ID[r["pred_label"]] for r in valid_rows])
    stats["accuracy"] = float(accuracy_score(y_true, y_pred))
    stats["macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))
    stats["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    stats["classification_report"] = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=["true", "false", "mixture"],
        output_dict=True,
        zero_division=0,
    )
    return stats


def evaluate_split(
    split_ds: Dataset,
    *,
    text_key: str,
    label_mapper,
    model: str,
    dataset_name: str,
    config: Dict,
) -> Tuple[List[Dict], Dict]:
    eval_ds = sample_or_full_split(split_ds, config.get("sample_size"), config.get("seed", 42))
    rows: List[Dict] = []
    started = time.perf_counter()
    iterable = tqdm(eval_ds, total=len(eval_ds), desc=f"{model} {dataset_name}")
    for idx, ex in enumerate(iterable):
        gold_label = label_mapper(ex)
        if gold_label is None:
            continue
        messages = build_prompt_messages(ex[text_key], config)
        per_item_start = time.perf_counter()
        pred_label, raw_response = safe_model_label(
            messages,
            model=model,
            retries=config.get("retry_on_invalid", 1),
            temperature=config.get("temperature", 0.0),
            max_tokens=config.get("max_tokens", 16),
            reasoning_effort=config.get("reasoning_effort"),
        )
        per_item_elapsed = time.perf_counter() - per_item_start
        rows.append(
            {
                "index": idx,
                "id": ex.get("id"),
                "text": ex[text_key],
                "gold_label": gold_label,
                "pred_label": pred_label,
                "raw_response": raw_response,
                "latency_sec": per_item_elapsed,
            }
        )
    elapsed = time.perf_counter() - started
    stats = compute_stats(rows, model=model, dataset_name=dataset_name, split=config.get("split", "test"), elapsed_sec=elapsed)
    return rows, stats


def slugify_model_name(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", model).strip("_")


def save_model_dataset_artifacts(base_dir: str, model: str, dataset_name: str, rows: List[Dict], stats: Dict) -> Dict[str, str]:
    model_slug = slugify_model_name(model)
    dataset_slug = dataset_name.lower().replace(" ", "_")
    model_dir = Path(base_dir) / model_slug
    model_dir.mkdir(parents=True, exist_ok=True)

    responses_path = model_dir / f"{dataset_slug}_responses.jsonl"
    with responses_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats_path = model_dir / f"{dataset_slug}_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    return {"responses": str(responses_path), "stats": str(stats_path)}


def save_run_summary(summary_path: str, summary_rows: List[Dict]) -> None:
    if not summary_rows:
        return
    fieldnames = [
        "model",
        "dataset",
        "split",
        "sample_count",
        "valid_count",
        "coverage",
        "accuracy",
        "macro_f1",
        "elapsed_sec",
        "timestamp_utc",
    ]
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

