# eval/eval_all.py
import argparse
import csv
import json
import sys
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

from eval.edit_distance import safe_edit_distance


LANGS_ALLOWED = ["en", "ar", "de", "ru", "sw", "vi", "zh", "bg", "el", "es", "hi", "th", "tr"]
TASKS_ALLOWED = ["all", "xnli", "sib200", "taxi1500"]
SPLITS_ALLOWED = ["all", "train", "test", "validation"]


# -----------------------------
# basic helpers
# -----------------------------
def infer_task_split_from_path(p: Path) -> Tuple[str, str]:
    split = p.parent.name
    task = p.parent.parent.name
    return task, split


def lang_from_filename(p: Path) -> Optional[str]:
    # supports zh.jsonl or zh_*.jsonl
    stem = p.name.split(".", 1)[0]
    pref = stem.split("_", 1)[0]
    return pref if pref in set(LANGS_ALLOWED) else None


def get_src_text(task: str, ex: Dict, lang: str) -> str:
    if task in ("sib200", "taxi1500"):
        src = ex.get("text", {}).get(lang, "")
    else:  # xnli
        src = ex.get("premise", {}).get(lang, "")
    return "" if src is None else str(src)



def get_orig_label_text(ex: Dict, lang: str) -> str:
    v = ex.get("orig_pred_text", {}).get(lang, "")
    return ("" if v is None else str(v)).strip().lower()


def get_target_label_text(task: str, ex: Dict, lang: str) -> str:
    v = ex.get("target_pred_text", {}).get(lang, None)
    if v is not None and str(v).strip():
        return str(v).strip().lower()

    target_id = ex.get("target_pred", {}).get(lang, None)
    if target_id is None:
        return ""
    target_id = int(target_id)

    if task == "xnli":
        labels = ["entailment", "neutral", "contradiction"]
    elif task == "sib200":
        labels = [
            "entertainment", "geography", "health", "politics",
            "science/technology", "sports", "travel",
        ]
    else:  # taxi1500
        labels = ["recommendation", "faith", "violence", "grace", "sin", "description"]

    if 0 <= target_id < len(labels):
        return labels[target_id].strip().lower()
    return ""


def ppl_of_text(model, tokenizer, text: str, device: torch.device, max_length: int = 512) -> float:
    text = "" if text is None else str(text)
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask", None)
    if attn is not None:
        attn = attn.to(device)
    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attn, labels=input_ids)
        return float(torch.exp(out.loss).detach().cpu().item())


@dataclass
class RunningStats:
    n_cfs: int = 0

    soft_sum: float = 0.0
    soft_n: int = 0

    hard_sum: float = 0.0
    hard_n: int = 0

    ppl_sum: float = 0.0
    ppl_n: int = 0

    sim_sum: float = 0.0
    sim_n: int = 0

    edit_sum: float = 0.0
    edit_n: int = 0

    def add_bool(self, which: str, b: Optional[bool]) -> None:
        if b is None:
            return
        v = 1.0 if bool(b) else 0.0
        if which == "soft":
            self.soft_sum += v
            self.soft_n += 1
        elif which == "hard":
            self.hard_sum += v
            self.hard_n += 1

    def add_float(self, which: str, x: Optional[float]) -> None:
        if x is None:
            return
        try:
            v = float(x)
        except Exception:
            return
        if v != v:
            return
        if which == "ppl":
            self.ppl_sum += v
            self.ppl_n += 1
        elif which == "sim":
            self.sim_sum += v
            self.sim_n += 1
        elif which == "edit":
            self.edit_sum += v
            self.edit_n += 1

    def mean(self, which: str) -> float:
        if which == "soft":
            return self.soft_sum / self.soft_n if self.soft_n else float("nan")
        if which == "hard":
            return self.hard_sum / self.hard_n if self.hard_n else float("nan")
        if which == "ppl":
            return self.ppl_sum / self.ppl_n if self.ppl_n else float("nan")
        if which == "sim":
            return self.sim_sum / self.sim_n if self.sim_n else float("nan")
        if which == "edit":
            return self.edit_sum / self.edit_n if self.edit_n else float("nan")
        return float("nan")


def compute_metrics_for_file(
    task: str,
    lang: str,
    path: Path,
    *,
    do_ppl: bool,
    do_sim: bool,
    do_edit: bool,
    ppl_model,
    ppl_tok,
    ppl_device: Optional[torch.device],
    ppl_max_length: int,
    sim_model: Optional[SentenceTransformer],
    sim_device: str,
    sim_batch_size: int,
) -> RunningStats:
    st = RunningStats()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            if isinstance(ex, dict) and "error" in ex:
                continue

            y_orig = get_orig_label_text(ex, lang)
            y_tgt = get_target_label_text(task, ex, lang)
            src_text = get_src_text(task, ex, lang)

            cfs = ex.get("counterfactual", {}).get(lang, [])
            if not isinstance(cfs, list) or len(cfs) == 0:
                continue
            


            src_emb = None
            if do_sim and sim_model is not None:
                src_emb = sim_model.encode(
                    [src_text],
                    convert_to_tensor=True,
                    device=sim_device,
                    batch_size=sim_batch_size,
                    show_progress_bar=False,
                )

            sim_texts: List[str] = []
            for cf in cfs:

                st.n_cfs += 1
                pred = ("" if cf.get("pred") is None else str(cf.get("pred"))).strip().lower()
                cf_text = "" if cf.get("text") is None else str(cf.get("text"))

                if y_orig:
                    st.add_bool("soft", pred != y_orig)
                if y_tgt:
                    st.add_bool("hard", pred == y_tgt)

                if do_ppl and ppl_model is not None:
                    st.add_float(
                        "ppl",
                        ppl_of_text(
                            ppl_model, ppl_tok, cf_text,
                            device=ppl_device,
                            max_length=ppl_max_length
                        ),
                    )

                if do_edit:
                    st.add_float("edit", safe_edit_distance(src_text, cf_text))

                if do_sim and sim_model is not None and src_emb is not None:
                    sim_texts.append(cf_text)
                    

            if do_sim and sim_model is not None and src_emb is not None and len(sim_texts) > 0:
                cf_embs = sim_model.encode(
                    sim_texts,
                    convert_to_tensor=True,
                    device=sim_device,
                    batch_size=sim_batch_size,
                    show_progress_bar=False,
                )
                sims = sim_model.similarity(src_emb, cf_embs)  # cosine similarity matrix [1,k]
                for j in range(int(sims.shape[1])):
                    st.add_float("sim", float(sims[0, j].item()))
                    
                    

    return st


def call_process_all_reward_as_library(args) -> None:
    """
    Call reward.process_all_reward.main() by temporarily replacing sys.argv.
    This reuses the existing script without copying its code.
    """
    import reward.process_all_rewards as proc

    argv = ["process_all_reward.py"]
    argv += ["--in_root", str(args.in_root)]
    argv += ["--out_root", str(args.reward_out_root)]
    argv += ["--task", str(args.task)]
    argv += ["--local_model_dir", str(args.local_model_dir)]
    #argv += ["--use_chat_template"]

    if args.langs != "all":
        argv += ["--langs"] + list(args.langs.split(","))

    old_argv = sys.argv
    try:
        sys.argv = argv
        proc.main()
    finally:
        sys.argv = old_argv


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--in_root", type=str, required=True)
    ap.add_argument("--reward_out_root", type=str, required=True)

    ap.add_argument("--run_rewards", action="store_true")
    ap.add_argument("--resume_rewards", action="store_true")
    ap.add_argument("--overwrite_rewards", action="store_true")
    ap.add_argument("--max_examples", type=int, default=None)

    ap.add_argument("--task", choices=TASKS_ALLOWED, default="all")
    ap.add_argument("--split", choices=SPLITS_ALLOWED, default="all")
    ap.add_argument("--langs", type=str, default="all")  # comma-separated or "all"

    ap.add_argument("--do_ppl", action="store_true")
    ap.add_argument("--do_sim", action="store_true")
    ap.add_argument("--do_edit", action="store_true")

    ap.add_argument("--output_csv", type=str, required=True)
    ap.add_argument("--overwrite_csv", action="store_true")

    # pass-through to reward.process_all_reward
    ap.add_argument("--local_model_dir", type=str, default="Qwen/Qwen3-8B")
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--use_chat_template", action="store_true")

    ap.add_argument("--nllb_model_name", type=str, default="facebook/nllb-200-distilled-1.3B")
    ap.add_argument("--do_align", action="store_true", default=True)

    # ppl args
    ap.add_argument("--ppl_model_name_or_path", type=str, default="ai-forever/mGPT")
    ap.add_argument("--ppl_device", type=str, default="cuda")
    ap.add_argument("--ppl_dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--ppl_max_length", type=int, default=512)

    # sim args
    ap.add_argument("--sim_model_name_or_path", type=str, default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    ap.add_argument("--sim_device", type=str, default="cuda")
    ap.add_argument("--sim_batch_size", type=int, default=128)

    args = ap.parse_args()

    in_root = Path(args.in_root).expanduser().resolve()
    reward_out_root = Path(args.reward_out_root).expanduser().resolve()

    # overwrite_rewards: remove existing outputs before running
    if args.overwrite_rewards and reward_out_root.exists():
        # conservative: only delete the filtered subtree if task != all
        if args.task == "all":
            shutil.rmtree(reward_out_root)
        else:
            sub = reward_out_root / args.task
            if sub.exists():
                shutil.rmtree(sub)

    # run rewards by calling existing script
    if args.run_rewards:
        reward_out_root.mkdir(parents=True, exist_ok=True)
        call_process_all_reward_as_library(args)

    # stage B: compute metrics from reward_out_root
    scan_root = reward_out_root if args.task == "all" else (reward_out_root / args.task)
    if not scan_root.exists():
        raise FileNotFoundError(f"reward_out_root not found: {scan_root}")

    langs_to_run = set(LANGS_ALLOWED) if args.langs == "all" else set(
        [x.strip() for x in args.langs.split(",") if x.strip()]
    )

    # load ppl/sim models
    ppl_model = ppl_tok = None
    ppl_device = None
    if args.do_ppl:
        if args.ppl_dtype == "float16":
            ppl_dtype = torch.float16
        elif args.ppl_dtype == "bfloat16":
            ppl_dtype = torch.bfloat16
        else:
            ppl_dtype = torch.float32
        ppl_device = torch.device(args.ppl_device)
        ppl_tok = AutoTokenizer.from_pretrained(args.ppl_model_name_or_path, local_files_only=args.local_files_only)
        ppl_model = AutoModelForCausalLM.from_pretrained(
            args.ppl_model_name_or_path,
            torch_dtype=ppl_dtype,
            local_files_only=args.local_files_only,
        ).to(ppl_device)
        ppl_model.eval()

    sim_model = SentenceTransformer(args.sim_model_name_or_path, device=args.sim_device) if args.do_sim else None

    # collect files
    reward_jsonl_paths = sorted(scan_root.rglob("*.jsonl"))
    metric_files: List[Tuple[Path, str, str, str]] = []
    for p in reward_jsonl_paths:
        task, split = infer_task_split_from_path(p)
        if task not in ["xnli", "sib200", "taxi1500"]:
            continue
        if args.task != "all" and task != args.task:
            continue
        if args.split != "all" and split != args.split:
            continue
        lang = lang_from_filename(p)
        if lang is None or lang not in langs_to_run:
            continue
        metric_files.append((p, task, split, lang))

    if not metric_files:
        raise RuntimeError(f"No reward jsonl files found under {scan_root} with filters.")

    out_csv = Path(args.output_csv).expanduser().resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.overwrite_csv and out_csv.exists():
        out_csv.unlink()

    fieldnames = [
        "task", "split", "lang",
        "soft_flip_rate", "hard_flip_rate",
        "ppl_mean", "similarity_mean", "edit_distance_mean",
        "n_counterfactuals",
        "soft_n", "hard_n", "ppl_n", "sim_n", "edit_n",
        "input_file",
    ]

    write_header = not out_csv.exists()
    with out_csv.open("a", newline="", encoding="utf-8") as fcsv:
        w = csv.DictWriter(fcsv, fieldnames=fieldnames)
        if write_header:
            w.writeheader()

        for p, task, split, lang in metric_files:
            st = compute_metrics_for_file(
                task=task,
                lang=lang,
                path=p,
                do_ppl=args.do_ppl,
                do_sim=args.do_sim,
                do_edit=args.do_edit,
                ppl_model=ppl_model,
                ppl_tok=ppl_tok,
                ppl_device=ppl_device,
                ppl_max_length=args.ppl_max_length,
                sim_model=sim_model,
                sim_device=args.sim_device,
                sim_batch_size=args.sim_batch_size,
            )
            w.writerow({
                "task": task,
                "split": split,
                "lang": lang,
                "soft_flip_rate": st.mean("soft"),
                "hard_flip_rate": st.mean("hard"),
                "ppl_mean": st.mean("ppl") if args.do_ppl else "",
                "similarity_mean": st.mean("sim") if args.do_sim else "",
                "edit_distance_mean": st.mean("edit") if args.do_edit else "",
                "n_counterfactuals": st.n_cfs,
                "soft_n": st.soft_n,
                "hard_n": st.hard_n,
                "ppl_n": st.ppl_n,
                "sim_n": st.sim_n,
                "edit_n": st.edit_n,
                "input_file": str(p),
            })
            fcsv.flush()

    print(f"[done] wrote csv: {out_csv}")


if __name__ == "__main__":
    main()
