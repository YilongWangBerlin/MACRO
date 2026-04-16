#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import pandas as pd


DEFAULT_ROOT = Path("/MACRO/general/llm_eval/outputs")
DEFAULT_OUT_CSV = Path("/MACRO/general/mmlu_accs.csv")


def read_jsonl_lines(fpath: str):
    rows = []
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def infer_model_dataset_from_jsonl_path(fpath: Path) -> dict:
    tokens = fpath.parts
    outs_idx = None
    for i, tok in enumerate(tokens):
        if tok == "outputs":
            outs_idx = i
            break
    else:
        raise ValueError("Path does not contain 'outputs/'")

    if outs_idx is None or len(tokens) <= outs_idx + 2:
        raise ValueError("Output path not in expected format.")

    model_name = tokens[outs_idx + 1]
    dataset_type = tokens[outs_idx + 2]
    file_name = tokens[-1]

    return {
        "model": model_name,
        "dataset_type": dataset_type,
        "jsonl_file": file_name,
    }


def main():
    if len(sys.argv) > 1:
        ROOT = Path(sys.argv[1])
    else:
        ROOT = DEFAULT_ROOT

    if len(sys.argv) > 2:
        OUT_CSV = Path(sys.argv[2])
    else:
        OUT_CSV = DEFAULT_OUT_CSV

    print(f"Scanning: {ROOT}")
    records = []

    for rootp in ROOT.iterdir():
        if not rootp.is_dir():
            continue

        for datap in rootp.iterdir():
            if not datap.is_dir():
                continue

            for jsonl_path in datap.glob("*.jsonl"):
                meta = infer_model_dataset_from_jsonl_path(jsonl_path)

                rows = read_jsonl_lines(str(jsonl_path))
                rows = [r for r in rows if "__is_correct" in r and "__source_idx" in r]

                if not rows:
                    continue

                n_total = len(rows)
                n_correct = sum(1 for r in rows if r["__is_correct"] is True)
                acc = n_correct / n_total if n_total > 0 else 0.0

                records.append({
                    "model": meta["model"],
                    "dataset_type": meta["dataset_type"],
                    "jsonl_file": meta["jsonl_file"],
                    "total": n_total,
                    "correct": n_correct,
                    "accuracy": acc,
                })

    df = pd.DataFrame(records)

    df["accuracy"] = df["accuracy"].round(6)

    df_group = (
        df.groupby(["model", "dataset_type"])["accuracy"]
        .agg(["mean", "std", "count"])
        .round(6)
        .reset_index()
    )
    df_group = df_group.rename(
        columns={
            "mean": "acc_mean",
            "std": "acc_std",
            "count": "n_files",
        }
    )


    df = df.merge(df_group, on=["model", "dataset_type"], how="left")

    df.to_csv(OUT_CSV, index=False)

    print(f"Saved summary to: {OUT_CSV}")
    print("\nPer-file accuracy:")
    print(df.head(10))

    print("\nPer-model-dataset accuracy:")
    print(df_group.head(10))


if __name__ == "__main__":
    main()