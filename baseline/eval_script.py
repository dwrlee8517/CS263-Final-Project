# %%
import os

# Logic to switch between colab and local IDE
main_dir = os.curdir

print(f"Main Directory: {main_dir}")

# Store the clean, downloaded BERT model here
base_model_path = os.path.join(main_dir, "model", "bertweet_base_local")
# Store your training checkpoints here
checkpoint_dir = os.path.join(main_dir, "model", "checkpoints")
# Store the datasets here
dataset_dir = os.path.join(main_dir, "data")

os.makedirs(base_model_path, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(dataset_dir, exist_ok=True)

print(f"Base model will be stored at: {base_model_path}")
print(f"Checkpoints will be saved to: {checkpoint_dir}")
print(f"Datasets will be stored to: {dataset_dir}")


# %% [markdown]
# ## Load Datasets

# %%
from datasets import load_dataset
from huggingface_hub import list_repo_files, hf_hub_download
import os

# Reuse dataset_dir from the setup cell
os.makedirs(dataset_dir, exist_ok=True)

revision = "refs/convert/parquet"
subsets = ["pubhealth_source", "pubhealth_bigbio_pairs"]

repo_files = list_repo_files(
    "bigbio/pubhealth",
    repo_type="dataset",
    revision=revision,
)

def download_split_files(subset_name):
    parquet_files = [
        f for f in repo_files
        if f.startswith(subset_name) and f.endswith(".parquet")
    ]
    if not parquet_files:
        raise RuntimeError(
            f"No parquet files found for subset {subset_name} at {revision}"
        )

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

    # Remove empty splits to avoid errors
    return {k: v for k, v in split_files.items() if v}

source_files = download_split_files("pubhealth_source")
print(f"Source files: {source_files}")
pair_files = download_split_files("pubhealth_bigbio_pairs")
print(f"Pair files: {pair_files}")

# Load each subset separately (different schemas)
source_ds = load_dataset("parquet", data_files=source_files)
pair_ds = load_dataset("parquet", data_files=pair_files)

index = 3
print(source_ds["train"][index])
print(pair_ds["train"][index])

# %% [markdown]
# ## Load Intel/misinformation-guard Dataset

# %%
import os

intel_ds = load_dataset("Intel/misinformation-guard")

int_to_text_map = {
    0: "false",
    1: "mostly_true",
    2: "partially_true",
    3: "true"
}

intel_ds = intel_ds.map(
    lambda x: {"label_text": int_to_text_map[x["label"]]}
)

intel_save_path = os.path.join(dataset_dir, "intel_misinformation_guard")
os.makedirs(intel_save_path, exist_ok=True)
intel_ds.save_to_disk(intel_save_path)
print(f"Intel dataset saved to: {intel_save_path}")

print(f"Intel dataset: {intel_ds['train'][1]}")

# %% [markdown]
# ## Setup API Environments

# %%

import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import Dict, List
import random
import re
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
deepinfra_api_key = os.getenv("DEEPINFRA_TOKEN")
gemini_api_key = os.getenv("GEMINI_API_KEY")
featherless_api_key = os.getenv("FEATHERLESS_API_KEY")

def generate_openai_response(prompt: str|List[Dict], *, openai_api_key=openai_api_key, **gen_params):
    client = OpenAI(api_key=openai_api_key)

    model = gen_params.get("model", "gpt-4o-mini")
    reasoning_value = gen_params.get("reasoning_effort") or gen_params.get("reasoning")
    if reasoning_value:
        reasoning_param = {"effort": reasoning_value}
    else:
        reasoning_param = None

    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]

    request_params = dict(gen_params)
    request_params.pop("model", None)
    request_params.pop("reasoning", None)
    request_params.pop("reasoning_effort", None)
    request_params.pop("temperature", None)
    request_params.pop("top_p", None)
    max_tokens = request_params.pop("max_tokens", None)
    # if max_tokens is not None:
    #     request_params["max_output_tokens"] = max_tokens

    response = client.responses.create(
        model=model,
        reasoning=reasoning_param,
        input=prompt,
        **request_params
    )
    return response.output_text

def generate_deepinfra_response(prompt: str|List[Dict], *, deepinfra_api_key=deepinfra_api_key, stream=False, **gen_params):
    if not deepinfra_api_key:
        raise ValueError("DEEPINFRA_TOKEN is missing. Set it in your environment.")

    client = OpenAI(api_key=deepinfra_api_key, base_url="https://api.deepinfra.com/v1/openai")

    model = gen_params.get("model", "meta-llama/Meta-Llama-3-8B-Instruct")
    temperature = gen_params.get("temperature", 0.0)
    max_tokens = gen_params.get("max_tokens", 16)

    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = prompt


    chat_completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
    )

    if stream:
        chunks = []
        for event in chat_completion:
            delta = event.choices[0].delta.content
            if delta:
                print(delta, end="", flush=True)
                chunks.append(delta)
        print()
        return "".join(chunks)

    return chat_completion.choices[0].message.content

def generate_gemini_response(prompt: str | List[Dict], *, gemini_api_key=gemini_api_key, stream=False, **gen_params):
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY is missing. Set it in your environment.")

    client = OpenAI(
        api_key=gemini_api_key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    model = gen_params.get("model", "models/gemini-3-flash-preview")
    reasoning_effort = gen_params.get("reasoning_effort")
    if reasoning_effort is None:
        reasoning_effort = gen_params.get("reasoning")

    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = prompt

    request_kwargs = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    if reasoning_effort:
        request_kwargs["reasoning_effort"] = reasoning_effort

    chat_completion = client.chat.completions.create(**request_kwargs)

    if stream:
        chunks = []
        for event in chat_completion:
            delta = event.choices[0].delta.content
            if delta:
                print(delta, end="", flush=True)
                chunks.append(delta)
        print()
        return "".join(chunks)

    return chat_completion.choices[0].message.content

def generate_featherless_response(prompt: str | List[Dict], *, featherless_api_key=featherless_api_key, stream=False, **gen_params):
    if not featherless_api_key:
        raise ValueError("FEATHERLESS_API_KEY is missing. Set it in your environment.")

    client = OpenAI(api_key=featherless_api_key, base_url="https://api.featherless.ai/v1")

    model = gen_params.get("model", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    temperature = gen_params.get("temperature", 0.0)
    max_tokens = gen_params.get("max_tokens", 16)

    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = prompt

    chat_completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
    )

    if stream:
        chunks = []
        for event in chat_completion:
            delta = event.choices[0].delta.content
            if delta:
                print(delta, end="", flush=True)
                chunks.append(delta)
        print()
        return "".join(chunks)

    return chat_completion.choices[0].message.content

def generate_model_response(prompt: str | List[Dict], **gen_params):
    model_name = str(gen_params.get("model", "")).lower()
    if "gpt" in model_name:
        return generate_openai_response(prompt, **gen_params)
    if "gemini" in model_name:
        return generate_gemini_response(prompt, stream=False, **gen_params)
    return generate_deepinfra_response(prompt, stream=False, **gen_params)


# %%
EVAL_LABEL_TO_ID = {"true": 0, "false": 1, "mixture": 2}
ID_TO_EVAL_LABEL = {v: k for k, v in EVAL_LABEL_TO_ID.items()}

# Map Intel labels into the 3-class PubHealth-compatible space.
INTEL_TO_EVAL_LABEL = {
    "true": "true",
    "false": "false",
    "mostly_true": "mixture",
    "partially_true": "mixture",
}
PUBHEALTH_TO_EVAL_LABEL = {
    0: "true",
    1: "false",
    2: None,  # unproven (skip in 3-class eval)
    3: "mixture",
}

MODEL_NAMES = [
    "gpt-5-nano",
    "models/gemini-2.5-flash-lite",
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
]

FINAL_CONFIG = {
    "sample_size": 10,
    "seed": 42,
    "split": "test",
    "temperature": 0.0,
    "max_tokens": 16,
    "retry_on_invalid": 1,
    "reasoning_effort": "low",
}


def build_zero_shot_messages(claim_text: str) -> List[Dict]:
    system_message = {
        "role": "system",
        "content": "You are a strict fact-checking assistant that only responds with labels."
    }
    user_message = {
        "role": "user",
        "content": (
            "Classify the following claim into exactly one of: true, false, mixture. "
            "Do not add any explanation, just output the label.\n\n"
            f"Claim: {claim_text}\n"
            "Label:"
        ),
    }
    return [system_message, user_message]


def parse_label(raw_text: str) -> str | None:
    if not raw_text:
        return None

    cleaned = raw_text.strip().lower()
    cleaned = re.sub(r"[^a-z_ ]", " ", cleaned)
    cleaned = " ".join(cleaned.split())

    if cleaned in EVAL_LABEL_TO_ID:
        return cleaned

    # Handle common variants first.
    if "partially true" in cleaned or "mostly true" in cleaned or "mixed" in cleaned:
        return "mixture"

    tokens = cleaned.split()
    for token in tokens:
        if token in EVAL_LABEL_TO_ID:
            return token
        if token == "partially_true" or token == "mostly_true":
            return "mixture"

    return None


def safe_model_label(messages: List[Dict], *, retries: int = 1, **kwargs) -> tuple[str | None, str]:
    last_raw = ""
    for _ in range(retries + 1):
        raw = generate_model_response(messages, **kwargs)
        last_raw = raw or ""
        parsed = parse_label(last_raw)
        if parsed is not None:
            return parsed, last_raw
    return None, last_raw


def sample_dataset(split_ds, sample_size: int, seed: int):
    n = len(split_ds)
    if sample_size >= n:
        return split_ds
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(n), sample_size))
    return split_ds.select(indices)


def summarize_eval_rows(rows, *, dataset_name: str, split: str, sample_size: int, model: str):
    valid_rows = [r for r in rows if r["pred_label"] is not None]
    coverage = len(valid_rows) / len(rows) if rows else 0.0

    if not valid_rows:
        print(f"No valid predictions were parsed for {dataset_name}.")
        print(f"Coverage: {coverage:.3f}")
        return

    y_true = np.array([EVAL_LABEL_TO_ID[r["gold_label"]] for r in valid_rows])
    y_pred = np.array([EVAL_LABEL_TO_ID[r["pred_label"]] for r in valid_rows])

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    print(f"\n=== Zero-shot Llama Evaluation ({dataset_name}) ===")
    print(f"Model: {model}")
    print(f"Split: {split}")
    print(f"Requested sample size: {sample_size}")
    print(f"Evaluated rows: {len(rows)}")
    print(f"Parsed coverage: {coverage:.3f} ({len(valid_rows)}/{len(rows)})")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print("\nConfusion Matrix (rows=true, cols=pred; [true,false,mixture]):")
    print(cm)
    print("\nClassification report:")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[0, 1, 2],
            target_names=["true", "false", "mixture"],
            digits=4,
            zero_division=0,
        )
    )

    mismatches = [r for r in valid_rows if r["gold_label"] != r["pred_label"]][:15]
    if mismatches:
        print("Sample mismatches (up to 15):")
        for i, row in enumerate(mismatches, 1):
            snippet = row["text"][:120].replace("\n", " ")
            print(
                f"{i}. gold={row['gold_label']} pred={row['pred_label']} "
                f"| text={snippet}..."
            )


def run_zero_shot_eval(dataset_split, *, text_key: str, label_mapper, dataset_name: str, config: Dict):
    sampled = sample_dataset(
        dataset_split,
        sample_size=config["sample_size"],
        seed=config["seed"],
    )

    rows = []
    for ex in sampled:
        gold_label = label_mapper(ex)
        if gold_label is None:
            continue

        messages = build_zero_shot_messages(ex[text_key])
        pred_label, raw_response = safe_model_label(
            messages,
            retries=config["retry_on_invalid"],
            model=config["model"],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            reasoning_effort=config.get("reasoning_effort"),
        )

        rows.append(
            {
                "id": ex.get("id"),
                "text": ex[text_key],
                "gold_label": gold_label,
                "pred_label": pred_label,
                "raw_response": raw_response,
            }
        )

    summarize_eval_rows(
        rows,
        dataset_name=dataset_name,
        split=config["split"],
        sample_size=config["sample_size"],
        model=config["model"],
    )

    return rows


all_results = {}
for model_name in MODEL_NAMES:
    model_config = dict(FINAL_CONFIG)
    model_config["model"] = model_name
    print("\n" + "=" * 80)
    print(f"Running evaluation for model: {model_name}")
    print("=" * 80)

    intel_rows = run_zero_shot_eval(
        intel_ds[model_config["split"]],
        text_key="text",
        label_mapper=lambda ex: INTEL_TO_EVAL_LABEL.get(ex["label_text"]),
        dataset_name=f"Intel sample ({model_name})",
        config=model_config,
    )

    pubhealth_rows = run_zero_shot_eval(
        source_ds[model_config["split"]],
        text_key="claim",
        label_mapper=lambda ex: PUBHEALTH_TO_EVAL_LABEL.get(ex["label"]),
        dataset_name=f"PubHealth sample ({model_name})",
        config=model_config,
    )

    all_results[model_name] = {
        "intel": intel_rows,
        "pubhealth": pubhealth_rows,
    }

# %%
