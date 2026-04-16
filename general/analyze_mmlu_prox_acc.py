#!/usr/bin/env python3
import json
import sys
from pathlib import Path
import pandas as pd


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


def infer_model_language_category_from_jsonl_path_and_row(row: dict, fpath: Path) -> dict:

    lang = None
    tokens = fpath.parts
    for i, tok in enumerate(tokens):
        if tok == "mmlu_prox":
            if i + 2 < len(tokens) and tokens[i + 1] == "json":
                name = tokens[i + 2]
                if name.endswith(".json"):
                    lang = name[:-5]
                break
    if not lang:
        in_json = row.get("__input_json", "")
        in_path = Path(in_json)
        if in_path.name.endswith(".json"):
            lang = in_path.name[:-5]

    model = None
    for i, tok in enumerate(tokens):
        if tok == "mmlu_prox":
            if i > 0:
                model = tokens[i - 1]
            break
    if not model:
        model = row.get("__model_name_or_path", row.get("__resolved_base_model", "unknown"))

    category = row.get("category", "unknown")

    return {
        "model": model,
        "language": lang,
        "category": category,
    }


def main():
    if len(sys.argv) > 1:
        ROOT = Path(sys.argv[1])
    else:
        ROOT = Path("MACRO/general/llm_eval/outputs")

    if len(sys.argv) > 2:
        OUT_CSV = Path(sys.argv[2])
    else:
        OUT_CSV = Path("MACRO/general/mmlu_prox_accs_lang_cat.csv")

    print(f"Scanning MMLU-ProX outputs: {ROOT}")
    records = []

    for model_dir in ROOT.iterdir():
        if not model_dir.is_dir():
            continue

        for dataset_subdir in model_dir.iterdir():
            if not dataset_subdir.is_dir() or dataset_subdir.name != "mmlu_prox":
                continue

            for jsonl_path in dataset_subdir.glob("*.jsonl"):
                rows = read_jsonl_lines(str(jsonl_path))

                rows = [r for r in rows if "__is_correct" in r and "__source_idx" in r]

                if not rows:
                    continue

                lang_meta = infer_model_language_category_from_jsonl_path_and_row(rows[0], jsonl_path)
                lang_model = lang_meta["model"]
                lang_lang = lang_meta["language"]

                n_total = len(rows)
                n_correct = sum(1 for r in rows if r["__is_correct"] is True)
                acc = n_correct / n_total if n_total > 0 else 0.0

                records.append({
                    "model": lang_model,
                    "language": lang_lang,
                    "category": "all",
                    "total": n_total,
                    "correct": n_correct,
                    "accuracy": acc,
                })

                from collections import defaultdict
                cat_stats = defaultdict(lambda: {"total": 0, "correct": 0})

                for r in rows:
                    cat = r.get("category", "unknown")
                    cat_stats[cat]["total"] += 1
                    if r["__is_correct"] is True:
                        cat_stats[cat]["correct"] += 1

                for cat, stat in cat_stats.items():
                    acc_cat = stat["correct"] / stat["total"] if stat["total"] > 0 else 0.0
                    records.append({
                        "model": lang_model,
                        "language": lang_lang,
                        "category": cat,
                        "total": stat["total"],
                        "correct": stat["correct"],
                        "accuracy": acc_cat,
                    })

    df = pd.DataFrame(records)
    df["accuracy"] = df["accuracy"].round(6)

    df.to_csv(OUT_CSV, index=False)

    print(f"Saved MMLU-ProX language+category accuracy summary to: {OUT_CSV}")
    print(df.head(20))


if __name__ == "__main__":
    main()