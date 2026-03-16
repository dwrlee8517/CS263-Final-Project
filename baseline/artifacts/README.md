# Evaluation Artifacts Reference

This directory stores the output from `evaluate.py` runs. Each run produces per-model and per-dataset logs plus a global summary so you can inspect responses, rerun comparisons, or import metrics into pandas/spreadsheets.

## Layout

- `run_config.json`: records the config file used, models evaluated, and paths (artifacts + dataset locations) for reproducibility.
- `runs_summary.csv` / `runs_summary.json`: tabular/nested summaries of every dataset/model pair in the current run.
- `<model_slug>/`: one subdirectory per model. Slugified versions of the model names (e.g., `meta-llama_Meta-Llama-3.1-8B-Instruct`).
  - `<dataset>_responses.jsonl`: newline-delimited JSON rows capturing each evaluated example:
    - `index`, `id`, `text`, `gold_label`, `pred_label`, `raw_response`, `latency_sec`
  - `<dataset>_stats.json`: aggregate metrics for that dataset (accuracy, macro-F1, coverage, confusion matrix, classification report, elapsed time, timestamp).

## Working with the files

1. **Inspect per-example responses**: use pandas or `jq` for `jsonl` files:
   ```sh
   python - <<'PY'
   import pandas as pd
   df = pd.read_json("artifacts/gpt-5-nano/intel_responses.jsonl", lines=True)
   print(df.head())
   PY
   ```

2. **Review stats**: read JSON to see the confusion matrix / classification report:
   ```python
   import json
   with open("artifacts/meta-llama_Meta-Llama-3.1-8B-Instruct/pubhealth_stats.json") as f:
       stats = json.load(f)
   print(stats["accuracy"], stats["macro_f1"])
   ```

3. **Compare runs**: open `runs_summary.csv` in Excel/Sheets or load with pandas:
   ```python
   import pandas as pd
   df = pd.read_csv("artifacts/runs_summary.csv")
   print(df.pivot(index="model", columns="dataset", values="macro_f1"))
   ```

4. **Trace the config**: `run_config.json` tells you which config file and parameters were used; keep a copy per run for auditability.

When you re-run `evaluate.py`, the artifacts directory will be reused; rename or archive older runs if you need multiple snapshots.
