
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import math
from datetime import datetime, timedelta

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from reward.reward_pred_aug import eval_one_example
from reward.reward_edit_lev import reward_edit
from reward.reward_align_nllb import load_nllb, nllb_ce_loss_and_reward, LANG2NLLB


LANGS_ALLOWED = ["en", "ar", "de", "ru", "sw", "vi", "zh", "bg", "el", "es", "hi", "th", "tr"]
TASKS_ALLOWED = ["all", "xnli", "sib200", "taxi1500"]

# NEW: tgt_lang (arg) -> NLLB language code, per user mapping
TGT_LANG2NLLB = {
    "en": "eng_Latn",
    "ar": "arb_Arab",
    "de": "deu_Latn",
    "ru": "rus_Cyrl",
    "sw": "swh_Latn",
    "vi": "vie_Latn",
    "zh": "zho_Hans",
    "bg": "Bulgarian",
    "el": "Greek",
    "es": "Spanish",
    "hi": "Hindi",
    "th": "Thai",
    "tr": "Turkish",
}


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def infer_task_split_from_path(p: Path) -> Tuple[str, str]:
    # expects: .../{task}/{split}/{file}.jsonl
    split = p.parent.name
    task = p.parent.parent.name
    return task, split


def find_lang_file(folder: Path, lang: str) -> Path:
    cands = sorted(folder.glob(f"{lang}_*.jsonl"))
    if not cands:
        raise FileNotFoundError(f"Cannot find {lang}_*.jsonl under {folder}")
    return cands[0]


def count_existing_lines(p: Path) -> int:
    if not p.exists():
        return 0
    with p.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def count_total_lines(p: Path) -> int:
    with p.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def fmt_finish_time_from_remaining(remaining_seconds: float) -> str:
    if remaining_seconds is None or not math.isfinite(float(remaining_seconds)):
        return "N/A"
    finish = datetime.now() + timedelta(seconds=float(remaining_seconds))
    return finish.strftime("%Y-%m-%d %H:%M:%S")


def under_task(in_root: Path, file_path: Path, task: str) -> bool:
    """
    Keep only files under in_root/{task}/... when task != 'all'.
    """
    if task == "all":
        return True
    try:
        rel = file_path.resolve().relative_to(in_root.resolve())
    except Exception:
        return False
    return len(rel.parts) >= 1 and rel.parts[0] == task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_root", type=str, default="./data_qwen_sample_no_min")
    ap.add_argument("--out_root", type=str, default="./data_qwen8_macro_reward_full_4")

    ap.add_argument(
        "--task",
        type=str,
        default="all",
        choices=TASKS_ALLOWED,
        help="Which task subtree to process: all/xnli/sib200/taxi1500.",
    )

    ap.add_argument(
        "--langs",
        nargs="+",
        default=None,
        choices=LANGS_ALLOWED,
        help="Only process these langs, e.g. --langs zh en. Default: all allowed langs.",
    )

    ap.add_argument("--local_model_dir", type=str, default="Qwen/Qwen3-8B")
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--use_chat_template", action="store_true")

    ap.add_argument("--nllb_model_name", type=str, default="facebook/nllb-200-distilled-1.3B")
    ap.add_argument("--do_align", action="store_true", default=True)

    # NEW: choose alignment target language (anchor language)
    ap.add_argument(
        "--tgt_lang",
        type=str,
        default="en",
        choices=LANGS_ALLOWED,
        help="Alignment anchor language (also mapped to NLLB FLORES code).",
    )

    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_examples", type=int, default=None)
    args = ap.parse_args()

    in_root = Path(args.in_root).expanduser().resolve()
    out_root = Path(args.out_root).expanduser().resolve()

    langs_to_run = set(args.langs) if args.langs is not None else set(LANGS_ALLOWED)

    if args.dtype == "float16":
        torch_dtype = torch.float16
    elif args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(
        args.local_model_dir,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.local_model_dir,
        device_map=args.device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )

    nllb_model = None
    nllb_tok = None
    if args.do_align:
        nllb_model, nllb_tok = load_nllb(args.nllb_model_name, device_map=args.device_map, dtype=args.dtype)

    if args.do_align and args.tgt_lang not in TGT_LANG2NLLB:
        raise ValueError(f"--tgt_lang {args.tgt_lang} not in TGT_LANG2NLLB mapping")

    # If task != all, restrict search to in_root/task to avoid scanning everything.
    scan_root = in_root if args.task == "all" else (in_root / args.task)
    if not scan_root.exists():
        raise FileNotFoundError(f"Task folder not found: {scan_root}")

    jsonl_paths = sorted(scan_root.rglob("*.jsonl"))

    eligible_paths: List[Path] = []
    global_total = 0
    for p in jsonl_paths:
        if not under_task(in_root, p, args.task):
            continue

        lang = p.name.split("_", 1)[0]
        if lang not in langs_to_run:
            continue

        task, split = infer_task_split_from_path(p)
        if args.task != "all" and task != args.task:
            continue

        out_path = out_root / task / split / p.name
        already = count_existing_lines(out_path) if args.resume else 0
        total_lines = count_total_lines(p)
        remaining = max(0, total_lines - already)
        if args.max_examples is not None:
            remaining = min(remaining, args.max_examples)
        if remaining <= 0:
            continue

        eligible_paths.append(p)
        global_total += remaining

    if global_total == 0:
        print("[done] nothing to process (all files are fully resumed or filtered out).")
        return

    global_pbar = tqdm(total=global_total, desc="ALL", unit="ex", dynamic_ncols=True)

    try:
        for p in eligible_paths:
            task, split = infer_task_split_from_path(p)
            if args.task != "all" and task != args.task:
                continue

            lang = p.name.split("_", 1)[0]
            if lang not in langs_to_run:
                continue

            out_path = out_root / task / split / p.name
            safe_mkdir(out_path.parent)

            already = count_existing_lines(out_path) if args.resume else 0
            total_lines = count_total_lines(p)

            remaining = max(0, total_lines - already)
            if args.max_examples is not None:
                remaining = min(remaining, args.max_examples)
            if remaining <= 0:
                continue

            # NEW: load anchor language file by --tgt_lang (instead of hard-coded English)
            anchor_by_index = None
            if args.do_align and lang != args.tgt_lang:
                try:
                    anchor_path = find_lang_file(p.parent, args.tgt_lang)
                    anchor_rows = read_jsonl(anchor_path)
                    anchor_by_index = {ex["index"]: ex for ex in anchor_rows}
                except Exception as e:
                    tqdm.write(f"[warn] align enabled but cannot load anchor={args.tgt_lang} for {p}: {e}")
                    anchor_by_index = None

            tqdm.write(
                f"[info] processing {task}/{split}/{p.name} -> {out_path} "
                f"(resume_skip={already}, todo={remaining})"
            )

            file_pbar = tqdm(
                total=remaining,
                desc=f"{task}/{split}/{lang}",
                unit="ex",
                dynamic_ncols=True,
                leave=False,
            )

            written = 0
            with p.open("r", encoding="utf-8") as rf, out_path.open("a", encoding="utf-8") as wf:
                for i, line in enumerate(rf):
                    if args.resume and i < already:
                        continue
                    if args.max_examples is not None and written >= args.max_examples:
                        break

                    try:
                        ex = json.loads(line)
                        
                        # cfs = ex.get("counterfactual", {}).get(lang, [])
                        # if isinstance(cfs, list) and len(cfs) > 0:
                        #     ex["counterfactual"][lang] = cfs[:3]
                        # else:
                        #     ex["counterfactual"][lang] = []

                        # 1) only process counterfactuals: pred/conf/flip/aug on each cf
                        eval_one_example(task, lang, ex, model, tokenizer, use_chat_template=args.use_chat_template)

                        # 2) edit reward
                        if task in ("sib200", "taxi1500"):
                            x = ex["text"][lang]
                        else:
                            x = ex["premise"][lang]
                        for cf in ex.get("counterfactual", {}).get(lang, []):
                            cf["reward_edit"] = float(reward_edit(cf["text"], x))

                        # 3) align reward (lang != tgt_lang) -> target anchor
                        if args.do_align and lang != args.tgt_lang and anchor_by_index is not None:
                            ex_anchor = anchor_by_index.get(ex["index"], None)
                            if ex_anchor is not None:
                                cfs_lang = ex.get("counterfactual", {}).get(lang, [])
                                cfs_anchor = ex_anchor.get("counterfactual", {}).get(args.tgt_lang, [])
                                n = min(len(cfs_lang), len(cfs_anchor))
                                if n > 0:
                                    src_texts = [cfs_lang[k]["text"] for k in range(n)]
                                    tgt_texts = [cfs_anchor[k]["text"] for k in range(n)]
                                    src_lang_code = LANG2NLLB[lang]
                                    tgt_lang_code = TGT_LANG2NLLB[args.tgt_lang]
                                    losses, rewards = nllb_ce_loss_and_reward(
                                        nllb_model,
                                        nllb_tok,
                                        src_texts,
                                        tgt_texts,
                                        src_lang=src_lang_code,
                                        tgt_lang=tgt_lang_code,
                                    )
                                    for k in range(n):
                                        cfs_lang[k]["reward_align"] = float(rewards[k])
                                        cfs_lang[k]["align_ce_loss"] = float(losses[k])

                        wf.write(json.dumps(ex, ensure_ascii=False) + "\n")
                        wf.flush()
                        written += 1

                    except Exception as e:
                        err_obj = {
                            "error": str(e),
                            "task": task,
                            "split": split,
                            "lang": lang,
                            "source_file": str(p),
                            "line_i": i,
                        }
                        wf.write(json.dumps(err_obj, ensure_ascii=False) + "\n")
                        wf.flush()
                        written += 1

                    file_pbar.update(1)
                    global_pbar.update(1)

                    rem = global_pbar.format_dict.get("remaining", None)
                    global_pbar.set_postfix_str(f"finish@{fmt_finish_time_from_remaining(rem)}")

            file_pbar.close()
            tqdm.write(f"[ok] wrote {written} new lines to {out_path}")

    finally:
        global_pbar.close()

    print("[done]")


if __name__ == "__main__":
    main()
