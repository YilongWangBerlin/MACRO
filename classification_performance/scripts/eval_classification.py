"""
Analysis: does DPO training change the model's original text classification
performance on SIB200 and TAXI1500?

We compare, for the exact same test-set inputs in the exact same language:
  (a) BEFORE: the base (pre-DPO) model's classification accuracy against gold
      labels -- already computed and stored in data_<model>_pred/<dataset>/test/
      (label = gold, orig_pred = base-model prediction), using
      preprocess/generate_predictions.py's own protocol (constrained decoding,
      greedy, temperature=0.0).
  (b) AFTER: we load the SAME base model with the paper's final MACRO LoRA
      adapter (the checkpoint used for the main results / Table 1) and
      classify the identical test text with the identical prompt/decoding
      protocol (imported from preprocess/generate_predictions.py, unmodified),
      then compare against the same gold labels.

This isolates the effect of DPO training on the model's own classification
behavior on ORIGINAL (unedited) text -- distinct from SLFR/hard-flip, which are
about candidate counterfactuals, and distinct from general-capability benchmarks
like MMLU.
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

from preprocess.generate_predictions import (  # noqa: E402 (reused, not modified)
    build_prompt_sib200, build_prompt_taxi1500,
    predict_label_constrained, SIB200_ID2LABEL, TAXI1500_ID2LABEL,
)

LANGS = ["en", "ar", "de", "ru", "sw", "vi", "zh"]
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ID2LABEL = {"sib200": SIB200_ID2LABEL, "taxi1500": TAXI1500_ID2LABEL}
BUILD_PROMPT = {"sib200": build_prompt_sib200, "taxi1500": build_prompt_taxi1500}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--adapter_path", type=str, required=True)
    ap.add_argument("--model_tag", type=str, required=True)
    ap.add_argument("--dataset", type=str, required=True, choices=["sib200", "taxi1500"])
    ap.add_argument("--pred_root", type=str, required=True, help="dir with BEFORE (base-model) predictions, for gold labels + baseline acc")
    ap.add_argument("--max_examples_per_lang", type=int, default=None)
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
        if args.max_examples_per_lang:
            examples = examples[: args.max_examples_per_lang]

        t_lang = time.time()
        n_correct_before = n_correct_after = 0
        n = 0
        for i, ex in enumerate(examples):
            text = ex["text"][lang]
            gold_id = int(ex["label"])
            gold = id2label[gold_id]
            before_pred = str(ex.get("orig_pred_text", {}).get(lang, "")).strip().lower()

            prompt = build_prompt(lang, text)
            # use_chat_template=False matches preprocess/generate_predictions.py's
            # main(), whose --use_chat_template flag defaults to False -- that is
            # the exact protocol that produced the BEFORE labels we compare against.
            out = predict_label_constrained(
                model=model, tokenizer=tokenizer, prompt_text=prompt, labels=labels,
                max_new_tokens=8, temperature=0.0, use_chat_template=False,
            )
            after_pred = out["label_text"]

            n += 1
            if before_pred == gold:
                n_correct_before += 1
            if after_pred == gold:
                n_correct_after += 1

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t_lang
                print(f"[{args.model_tag}/{args.dataset}/{lang}] {i+1}/{len(examples)} "
                      f"({elapsed/(i+1):.2f}s/ex)", flush=True)

        acc_before = 100.0 * n_correct_before / n if n else float("nan")
        acc_after = 100.0 * n_correct_after / n if n else float("nan")
        row = {
            "model": args.model_tag, "dataset": args.dataset, "lang": lang, "n": n,
            "acc_before_pct": round(acc_before, 1), "acc_after_pct": round(acc_after, 1),
            "delta_pct": round(acc_after - acc_before, 1),
        }
        rows.append(row)
        print(f"[done {lang}] {row}  ({(time.time()-t_lang)/60:.1f} min)", flush=True)

    out_csv = RESULTS_DIR / f"{args.model_tag}_{args.dataset}_classification_before_after.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] wrote {out_csv}", flush=True)
    print(f"[all done] total {(time.time()-t0)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
