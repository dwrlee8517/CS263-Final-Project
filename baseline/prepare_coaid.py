#!/usr/bin/env python3
"""
Prepare CoAID for this project's evaluation pipeline.

What this script does:
1) Loads CoAID CSV files across all released snapshots.
2) Harmonizes labels to the project's 3-class space by mapping:
      real -> true
      fake -> false
   (CoAID does not provide a native "mixture" class.)
3) Deduplicates rows by normalized text.
4) Creates stratified train/validation/test splits.
5) Writes parquet files that can be loaded with datasets.load_dataset("parquet", ...).
6) Prints a compact analysis summary and saves it as JSON.

Output columns:
  - id
  - text
  - label_text   ("true" | "false")
  - label        (3 for true, 0 for false; compatible with project conventions)
  - source_type  ("news" | "claim")
  - release      (e.g., "05-01-2020")
  - publish_date
  - fact_check_url
  - news_url
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


RELEASE_DIR_PATTERN = re.compile(r"^\d{2}-\d{2}-\d{4}$")
REAL_FILES = ("NewsRealCOVID-19.csv", "ClaimRealCOVID-19.csv")
FAKE_FILES = ("NewsFakeCOVID-19.csv", "ClaimFakeCOVID-19.csv")
LABEL_TO_INT = {"false": 0, "true": 3}


@dataclass(frozen=True)
class LoadedFrame:
    frame: pd.DataFrame
    release: str
    file_name: str
    source_type: str
    label_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CoAID data for evaluation pipeline.")
    parser.add_argument(
        "--coaid-root",
        type=Path,
        default=Path("third_party/CoAID"),
        help="Path to local CoAID repo root.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/coaid_harmonized"),
        help="Directory where parquet splits and analysis will be saved.",
    )
    parser.add_argument(
        "--include-claims",
        action="store_true",
        help="Include Claim*.csv rows in addition to News*.csv rows.",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        default=True,
        help="Deduplicate by normalized text (enabled by default).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic splits.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio. Test ratio is 1-train-val.",
    )
    parser.add_argument(
        "--clone-if-missing",
        action="store_true",
        help="Clone CoAID from GitHub if --coaid-root does not exist.",
    )
    return parser.parse_args()


def maybe_clone_repo(coaid_root: Path) -> None:
    if coaid_root.exists():
        return
    coaid_root.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/cuilimeng/CoAID.git",
            str(coaid_root),
        ],
        check=True,
    )


def normalize_text_for_dedupe(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def pick_text_column(row: pd.Series) -> str:
    # Prefer concise "claim-like" fields first, then longer fields.
    candidates = [
        row.get("title"),
        row.get("newstitle"),
        row.get("abstract"),
        row.get("content"),
    ]
    for value in candidates:
        if pd.isna(value):
            continue
        s = str(value).strip()
        if s:
            return s
    return ""


def iter_release_dirs(coaid_root: Path) -> Iterable[Path]:
    for p in sorted(coaid_root.iterdir()):
        if p.is_dir() and RELEASE_DIR_PATTERN.match(p.name):
            yield p


def load_release_file(path: Path, release: str, file_name: str, label_text: str) -> LoadedFrame:
    df = pd.read_csv(path)
    source_type = "claim" if file_name.lower().startswith("claim") else "news"
    return LoadedFrame(frame=df, release=release, file_name=file_name, source_type=source_type, label_text=label_text)


def collect_raw_frames(coaid_root: Path, include_claims: bool) -> List[LoadedFrame]:
    loaded: List[LoadedFrame] = []
    for release_dir in iter_release_dirs(coaid_root):
        release = release_dir.name
        files_to_load = [("true", REAL_FILES[0]), ("false", FAKE_FILES[0])]
        if include_claims:
            files_to_load.extend([("true", REAL_FILES[1]), ("false", FAKE_FILES[1])])
        for label_text, file_name in files_to_load:
            f = release_dir / file_name
            if not f.exists():
                continue
            loaded.append(load_release_file(f, release, file_name, label_text))
    if not loaded:
        raise RuntimeError(f"No CoAID CSV files found under {coaid_root}.")
    return loaded


def to_unified_frame(loaded_frames: List[LoadedFrame]) -> pd.DataFrame:
    rows: List[Dict] = []
    for item in loaded_frames:
        df = item.frame
        for i, row in df.iterrows():
            text = pick_text_column(row)
            if not text:
                continue
            rows.append(
                {
                    "id": f"{item.release}:{item.file_name}:{i}",
                    "text": text,
                    "label_text": item.label_text,
                    "label": LABEL_TO_INT[item.label_text],
                    "source_type": item.source_type,
                    "release": item.release,
                    "publish_date": row.get("publish_date"),
                    "fact_check_url": row.get("fact_check_url"),
                    "news_url": row.get("news_url"),
                }
            )
    unified = pd.DataFrame(rows)
    if unified.empty:
        raise RuntimeError("No rows with usable text were produced from CoAID input.")
    return unified


def stratified_split(df: pd.DataFrame, train_ratio: float, val_ratio: float, seed: int) -> Dict[str, pd.DataFrame]:
    if not (0.0 < train_ratio < 1.0):
        raise ValueError("--train-ratio must be between 0 and 1.")
    if not (0.0 <= val_ratio < 1.0):
        raise ValueError("--val-ratio must be between 0 and 1.")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0.")

    test_ratio = 1.0 - train_ratio - val_ratio
    grouped = []
    for label, g in df.groupby("label_text", sort=True):
        g = g.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        n = len(g)
        n_train = int(round(n * train_ratio))
        n_val = int(round(n * val_ratio))
        if n_train + n_val > n:
            n_val = max(0, n - n_train)
        n_test = n - n_train - n_val
        # Keep at least one sample for test if possible.
        if test_ratio > 0 and n_test == 0 and n >= 3:
            n_test = 1
            if n_val > 0:
                n_val -= 1
            else:
                n_train -= 1
        grouped.append(
            (
                g.iloc[:n_train],
                g.iloc[n_train : n_train + n_val],
                g.iloc[n_train + n_val :],
                label,
            )
        )

    train = pd.concat([x[0] for x in grouped], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val = pd.concat([x[1] for x in grouped], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test = pd.concat([x[2] for x in grouped], ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return {"train": train, "validation": val, "test": test}


def summarize(df: pd.DataFrame, splits: Dict[str, pd.DataFrame], deduped_rows: int) -> Dict:
    text_lengths = df["text"].astype(str).str.len()
    release_counts = df["release"].value_counts().sort_index().to_dict()
    source_type_counts = df["source_type"].value_counts().sort_index().to_dict()
    label_counts = df["label_text"].value_counts().sort_index().to_dict()

    split_summary = {}
    for split_name, split_df in splits.items():
        split_summary[split_name] = {
            "rows": int(len(split_df)),
            "label_counts": split_df["label_text"].value_counts().sort_index().to_dict(),
        }

    return {
        "total_rows_after_filtering": int(len(df)),
        "rows_removed_by_deduplication": int(deduped_rows),
        "label_counts": label_counts,
        "source_type_counts": source_type_counts,
        "release_counts": release_counts,
        "text_length_chars": {
            "min": int(text_lengths.min()),
            "p50": float(text_lengths.quantile(0.5)),
            "p90": float(text_lengths.quantile(0.9)),
            "max": int(text_lengths.max()),
            "mean": float(text_lengths.mean()),
        },
        "splits": split_summary,
    }


def print_summary(summary: Dict) -> None:
    print("\n=== CoAID Prepared Dataset Summary ===")
    print(f"Rows (post-filter): {summary['total_rows_after_filtering']}")
    print(f"Rows removed by dedupe: {summary['rows_removed_by_deduplication']}")
    print(f"Label counts: {summary['label_counts']}")
    print(f"Source counts: {summary['source_type_counts']}")
    print(f"Release counts: {summary['release_counts']}")
    print(f"Text length stats (chars): {summary['text_length_chars']}")
    print("Split sizes:")
    for split, stats in summary["splits"].items():
        print(f"  - {split}: {stats['rows']} rows | labels={stats['label_counts']}")


def main() -> None:
    args = parse_args()
    if args.clone_if_missing:
        maybe_clone_repo(args.coaid_root)
    if not args.coaid_root.exists():
        raise FileNotFoundError(
            f"CoAID root not found at {args.coaid_root}. "
            "Pass --clone-if-missing or set --coaid-root to a valid local clone."
        )

    loaded = collect_raw_frames(args.coaid_root, include_claims=args.include_claims)
    df = to_unified_frame(loaded)

    removed = 0
    if args.dedupe:
        before = len(df)
        df["_dedupe_text"] = df["text"].astype(str).map(normalize_text_for_dedupe)
        df = df.drop_duplicates(subset=["_dedupe_text"]).drop(columns=["_dedupe_text"]).reset_index(drop=True)
        removed = before - len(df)

    splits = stratified_split(
        df=df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, split_df in splits.items():
        out_path = args.output_dir / f"{split_name}.parquet"
        split_df.to_parquet(out_path, index=False)

    summary = summarize(df=df, splits=splits, deduped_rows=removed)
    summary["config"] = {
        "coaid_root": str(args.coaid_root),
        "output_dir": str(args.output_dir),
        "include_claims": args.include_claims,
        "dedupe": args.dedupe,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
    }
    summary_path = args.output_dir / "analysis_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print_summary(summary)
    print(f"\nSaved parquet splits to: {args.output_dir}")
    print(f"Saved analysis summary to: {summary_path}")
    print("\nLoad in pipeline with something like:")
    print(
        "  load_dataset('parquet', data_files={'train':'.../train.parquet',"
        " 'validation':'.../validation.parquet', 'test':'.../test.parquet'})"
    )


if __name__ == "__main__":
    main()
