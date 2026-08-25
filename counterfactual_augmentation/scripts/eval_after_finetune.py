"""
Evaluates a LoRA-fine-tuned classifier checkpoint (from train_classifier_lora.py)
on the SIB200/TAXI1500 test set, using the exact same protocol as
classification_performance/scripts/eval_classification.py (imported
functions from preprocess/generate_predictions.py, use_chat_template=False,
greedy constrained decoding) so all accuracy numbers here are
directly comparable.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from preprocess.generate_predictions import (  # noqa: E402
    build_prompt_sib200, build_prompt_taxi1500,
    predict_label_constrained, SIB200_ID2LABEL, TAXI1500_ID2LABEL,
)

LANGS = ["en", "ar", "de", "ru", "sw", "vi", "zh"]
ID2LABEL = {"sib200": SIB200_ID2LABEL, "taxi1500": TAXI1500_ID2LABEL}
BUILD_PROMPT = {"sib200": build_prompt_sib200, "taxi1500": build_prompt_taxi1500}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--adapter_path", type=str, required=True)
    ap.add_argument("--model_tag", type=str, required=True)
    ap.add_argument("--condition", type=str, required=True, help="e.g. baseline_sft / augmented_sft")
    ap.add_argument("--dataset", type=str, required=True, choices=["sib200", "taxi1500"])
    ap.add_argument("--pred_root", type=str, required=True)
    ap.add_argument("--out_csv", type=str, required=True)
    args = ap.parse_args()

    id2label = ID2LABEL[args.dataset]
    labels = [id2label[i] for i in range(len(id2label))]
    build_prompt = BUILD_PROMPT[args.dataset]

    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, device_map="balanced", trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()
    print(f"[info] base+adapter loaded in {time.time()-t0:.1f}s", flush=True)

    rows = []
    for lang in LANGS:
        in_file = Path(args.pred_root) / args.dataset / "test" / f"{lang}.jsonl"
        if not in_file.exists():
            print(f"[warn] missing {in_file}, skipping", flush=True)
            continue
        examples = [json.loads(l) for l in in_file.read_text(encoding="utf-8").splitlines() if l.strip()]

        t_lang = time.time()
        n_correct = 0
        for i, ex in enumerate(examples):
            text = ex["text"][lang]
            gold = id2label[int(ex["label"])]
            prompt = build_prompt(lang, text)
            out = predict_label_constrained(
                model=model, tokenizer=tokenizer, prompt_text=prompt, labels=labels,
                max_new_tokens=8, temperature=0.0, use_chat_template=False,
            )
            if out["label_text"] == gold:
                n_correct += 1
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t_lang
                print(f"[{args.model_tag}/{args.condition}/{args.dataset}/{lang}] {i+1}/{len(examples)} "
                      f"({elapsed/(i+1):.2f}s/ex)", flush=True)

        n = len(examples)
        acc = 100.0 * n_correct / n if n else float("nan")
        row = {"model": args.model_tag, "condition": args.condition, "dataset": args.dataset,
               "lang": lang, "n": n, "n_correct": n_correct, "acc_pct": round(acc, 1)}
        rows.append(row)
        print(f"[done {lang}] {row}  ({(time.time()-t_lang)/60:.1f} min)", flush=True)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] wrote {out_path}", flush=True)
    print(f"[all done] total {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
