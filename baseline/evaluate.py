import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from evaluation_pipeline import (
    FINAL_CONFIG,
    INTEL_TO_EVAL_LABEL,
    MODEL_NAMES,
    PUBHEALTH_TO_EVAL_LABEL,
    download_pubhealth_splits,
    evaluate_split,
    load_intel_dataset,
    load_project_paths,
    save_model_dataset_artifacts,
    save_run_summary,
)


def load_yaml_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise ImportError(
            "YAML config requested, but PyYAML is not installed. "
            "Install with: pip install pyyaml"
        ) from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Top-level YAML config must be a mapping/object.")
    return data


def parse_args():
    parser = argparse.ArgumentParser(description="Run full evaluation over Intel and PubHealth test splits.")
    parser.add_argument(
        "--config",
        default="evaluate_config.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Optional list of models to evaluate. Overrides YAML/defaults.",
    )
    parser.add_argument(
        "--split",
        default=None,
        choices=["train", "validation", "test"],
        help="Split to evaluate. Overrides YAML/defaults.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional sample size for debugging/full override. Overrides YAML/defaults.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Directory where artifacts will be written. Overrides YAML/defaults.",
    )
    parser.add_argument(
        "--main-dir",
        default=None,
        help="Project root used to resolve model/data paths. Overrides YAML/defaults.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    yaml_cfg = load_yaml_config(args.config)

    yaml_models = yaml_cfg.get("models")
    models = args.models if args.models else (yaml_models if yaml_models else MODEL_NAMES)

    config = dict(FINAL_CONFIG)
    yaml_eval_cfg = yaml_cfg.get("config", {})
    if not isinstance(yaml_eval_cfg, dict):
        raise ValueError("'config' in YAML must be a mapping/object.")
    config.update(yaml_eval_cfg)
    if args.split is not None:
        config["split"] = args.split
    if args.sample_size is not None:
        config["sample_size"] = args.sample_size

    yaml_paths = yaml_cfg.get("paths", {})
    if yaml_paths and not isinstance(yaml_paths, dict):
        raise ValueError("'paths' in YAML must be a mapping/object.")
    main_dir = args.main_dir if args.main_dir is not None else yaml_paths.get("main_dir", ".")
    artifacts_dir_value = (
        args.artifacts_dir if args.artifacts_dir is not None else yaml_paths.get("artifacts_dir", "artifacts")
    )

    paths = load_project_paths(main_dir)
    dataset_dir = paths["dataset_dir"]
    source_ds, _ = download_pubhealth_splits(dataset_dir=dataset_dir)
    intel_ds = load_intel_dataset(dataset_dir=dataset_dir)

    artifacts_root = Path(artifacts_dir_value)
    artifacts_root.mkdir(parents=True, exist_ok=True)
    run_dir_name = datetime.now(timezone.utc).strftime("run_%Y%m%d_%H%M%S")
    artifacts_dir = artifacts_root / run_dir_name
    artifacts_dir.mkdir(parents=True, exist_ok=False)
    run_meta = {
        "config_file": args.config,
        "models": models,
        "config": config,
        "paths": {
            "main_dir": main_dir,
            "artifacts_root_dir": str(artifacts_root),
            "artifacts_dir": str(artifacts_dir),
        },
        "dataset_dir": dataset_dir,
    }
    with (artifacts_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    print("\n" + "=" * 80)
    print("Run configuration")
    print("=" * 80)
    print(json.dumps(run_meta, indent=2))
    print("=" * 80)

    summary_rows = []

    for model_name in models:
        print("\n" + "=" * 80)
        print(f"Running model: {model_name}")
        print("=" * 80)

        intel_rows, intel_stats = evaluate_split(
            intel_ds[config["split"]],
            text_key="text",
            label_mapper=lambda ex: INTEL_TO_EVAL_LABEL.get(ex["label_text"]),
            model=model_name,
            dataset_name="intel",
            config=config,
        )
        intel_paths = save_model_dataset_artifacts(
            base_dir=str(artifacts_dir),
            model=model_name,
            dataset_name="intel",
            rows=intel_rows,
            stats=intel_stats,
        )
        print(
            f"Intel | acc={intel_stats['accuracy']} f1={intel_stats['macro_f1']} "
            f"coverage={intel_stats['coverage']:.3f} time={intel_stats['elapsed_sec']:.2f}s"
        )
        print(f"Saved: {intel_paths['responses']} and {intel_paths['stats']}")
        summary_rows.append(intel_stats)

        pub_rows, pub_stats = evaluate_split(
            source_ds[config["split"]],
            text_key="claim",
            label_mapper=lambda ex: PUBHEALTH_TO_EVAL_LABEL.get(ex["label"]),
            model=model_name,
            dataset_name="pubhealth",
            config=config,
        )
        pub_paths = save_model_dataset_artifacts(
            base_dir=str(artifacts_dir),
            model=model_name,
            dataset_name="pubhealth",
            rows=pub_rows,
            stats=pub_stats,
        )
        print(
            f"PubHealth | acc={pub_stats['accuracy']} f1={pub_stats['macro_f1']} "
            f"coverage={pub_stats['coverage']:.3f} time={pub_stats['elapsed_sec']:.2f}s"
        )
        print(f"Saved: {pub_paths['responses']} and {pub_paths['stats']}")
        summary_rows.append(pub_stats)

    save_run_summary(str(artifacts_dir / "runs_summary.csv"), summary_rows)
    with (artifacts_dir / "runs_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2)
    print(f"\nRun summary saved to {artifacts_dir / 'runs_summary.csv'}")
    print(f"Run summary JSON saved to {artifacts_dir / 'runs_summary.json'}")


if __name__ == "__main__":
    main()

