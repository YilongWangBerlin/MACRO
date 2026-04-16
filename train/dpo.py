
#!/usr/bin/env python3
import argparse
import datetime as _datetime
import json
import os
import random
import math
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import yaml
import wandb
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from torch.utils.data import DataLoader

from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, DPOConfig, SFTTrainer, SFTConfig

# Keep a short alias for timestamp generation
datetime = _datetime.datetime


def read_jsonl(path: Path):
    """Streaming JSONL reader."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def parse_kv_list(items: List[str]) -> Dict[str, int]:
    """Parse list like ['en=1000','de=2000'] into dict."""
    out: Dict[str, int] = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"Bad kv item: {it}, expected like en=1000")
        k, v = it.split("=", 1)
        out[k.strip()] = int(v.strip())
    return out


def ensure_dpo_text_columns(ds: Dataset) -> Dataset:
    """
    TRL DPOTrainer expects text columns: prompt/chosen/rejected.
    If we load a cached dataset from disk, these columns might be missing.
    We reconstruct them from ds['meta'] when possible.
    """
    cols = set(ds.column_names)
    if "meta" not in cols:
        return ds

    metas = ds["meta"]

    def _from_meta(key: str):
        return [(m or {}).get(key, None) for m in metas]

    if "prompt" not in cols:
        ds = ds.add_column("prompt", _from_meta("prompt"))
    if "chosen" not in cols:
        ds = ds.add_column("chosen", _from_meta("chosen"))
    if "rejected" not in cols:
        ds = ds.add_column("rejected", _from_meta("rejected"))

    return ds


def resolve_lora_target_modules(scope: str, custom: Optional[List[str]]) -> List[str]:
    """
    Map a high-level LoRA scope to target_modules names commonly used in
    LLaMA/Qwen-style architectures.
    """
    scope = (scope or "all").lower()

    if scope == "qv":
        return ["q_proj", "v_proj"]
    if scope == "attn":
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    if scope == "mlp":
        return ["gate_proj", "up_proj", "down_proj"]
    if scope == "all":
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if scope == "custom":
        if not custom:
            raise ValueError("lora_scope=custom requires --lora_target_modules ...")
        return list(custom)

    raise ValueError(f"Unknown lora_scope: {scope}")


def build_sft_dataset_from_dpo(ds: Dataset, only_flip: bool = True) -> Dataset:
    """
    Build SFT dataset from DPO triples by using prompt+chosen as supervised target.
    Filter to only chosen with reward_flip == 1.0 when only_flip=True.
    """
    ds = ensure_dpo_text_columns(ds)

    if "meta" in ds.column_names:

        def _is_flip(ex):
            m = ex.get("meta", {}) or {}
            rf = m.get("reward_flip_raw", None)
            if rf is None:
                rf = m.get("reward_flip", None)
            return (not only_flip) or (rf == 1.0)

        sft = ds.filter(_is_flip)
    else:
        sft = ds

    def _to_text(ex):
        p = ex.get("prompt", "") or ""
        c = ex.get("chosen", "") or ""
        return {"text": p + "\n" + c}

    remove_cols = [c for c in sft.column_names if c in ("prompt", "chosen", "rejected", "meta")]
    sft = sft.map(_to_text, remove_columns=remove_cols)
    return sft


class LanguageRoundRobinBatchSampler:
    """
    Yield batches in a fixed language order:
      en batch -> ar batch -> ... -> zh batch -> repeat.
    """
    def __init__(
        self,
        dataset: Dataset,
        languages: List[str],
        batch_size: int,
        seed: int = 42,
        shuffle_within_lang: bool = True,
        drop_last: bool = True,
        cycle: bool = True,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        self.dataset = dataset
        self.languages = list(languages)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle_within_lang = bool(shuffle_within_lang)
        self.drop_last = bool(drop_last)
        self.cycle = bool(cycle)

        metas = dataset["meta"]
        self.idx_by_lang: Dict[str, List[int]] = {l: [] for l in self.languages}
        for i, m in enumerate(metas):
            lang = (m or {}).get("language", None)
            if lang in self.idx_by_lang:
                self.idx_by_lang[lang].append(i)

        for l in self.languages:
            if len(self.idx_by_lang[l]) == 0:
                raise ValueError(f"No samples found for language={l}")

        if self.drop_last:
            for l in self.languages:
                if len(self.idx_by_lang[l]) < self.batch_size:
                    raise ValueError(
                        f"Language={l} has only {len(self.idx_by_lang[l])} samples, "
                        f"but batch_size={self.batch_size} and drop_last=True."
                    )

        self.lang2idx = {l: i for i, l in enumerate(self.languages)}

        rng = random.Random(self.seed)
        if self.shuffle_within_lang:
            for l in self.languages:
                rng.shuffle(self.idx_by_lang[l])

        def n_batches(n):
            return (n // self.batch_size) if self.drop_last else math.ceil(n / self.batch_size)

        per_lang_batches = [n_batches(len(self.idx_by_lang[l])) for l in self.languages]
        self.cycles = max(per_lang_batches) if self.cycle else min(per_lang_batches)

        self.ptr = {l: 0 for l in self.languages}
        self._epoch = 0

    def __len__(self):
        return self.cycles * len(self.languages)

    def _reshuffle_lang(self, lang: str):
        if not self.shuffle_within_lang:
            return
        lang_i = self.lang2idx[lang]
        rng = random.Random(self.seed + 10007 * self._epoch + 97 * lang_i)
        rng.shuffle(self.idx_by_lang[lang])

    def __iter__(self):
        self._epoch += 1
        for _ in range(self.cycles):
            for lang in self.languages:
                idxs = self.idx_by_lang[lang]
                start = self.ptr[lang]
                end = start + self.batch_size

                if end <= len(idxs):
                    batch = idxs[start:end]
                    self.ptr[lang] = end
                    yield batch
                    continue

                if not self.cycle:
                    return

                self.ptr[lang] = 0
                self._reshuffle_lang(lang)
                idxs = self.idx_by_lang[lang]

                if self.drop_last and len(idxs) < self.batch_size:
                    return

                batch = idxs[0:self.batch_size]
                self.ptr[lang] = self.batch_size
                yield batch


class MetaDPOTrainer(DPOTrainer):
    """
    Attach per-batch meta info into inputs['meta'] via a wrapped data_collator,
    then pop it here to avoid passing unknown keys downstream.

    Also supports custom batch_sampler by overriding get_train_dataloader().
    """
    def __init__(self, *args, rr_batch_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._rr_batch_sampler = rr_batch_sampler

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        self._last_meta = inputs.pop("meta", None)
        return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

    def get_train_dataloader(self):
        if self._rr_batch_sampler is None:
            return super().get_train_dataloader()

        dl = DataLoader(
            self.train_dataset,
            batch_sampler=self._rr_batch_sampler,
            collate_fn=self.data_collator,
            num_workers=getattr(self.args, "dataloader_num_workers", 0),
            pin_memory=getattr(self.args, "dataloader_pin_memory", True),
        )
        accel = getattr(self, "accelerator", None)
        return accel.prepare(dl) if accel is not None else dl



class StepFileLoggerCallback(TrainerCallback):
    """Write one JSONL sample per language per logging step + a global loss CSV."""
    def __init__(self, step_log_root: Path, task: str, languages: List[str], run_cfg: Dict[str, Any]):
        self.step_log_root = step_log_root
        self.task = task
        self.languages = languages
        self.run_cfg = run_cfg
        self.trainer = None  # assigned after trainer is built

        self._fp: Dict[str, Any] = {}
        self._loss_csv = None

    def on_train_begin(self, args, state, control, **kwargs):
        for lang in self.languages:
            p = self.step_log_root / self.task
            p.mkdir(parents=True, exist_ok=True)
            fp = (p / f"{lang}.jsonl").open("a", encoding="utf-8")
            self._fp[lang] = fp

        self.step_log_root.mkdir(parents=True, exist_ok=True)
        loss_csv = self.step_log_root / f"{self.task}_loss.csv"
        is_new = not loss_csv.exists()
        self._loss_csv = loss_csv.open("a", encoding="utf-8")
        if is_new:
            self._loss_csv.write("step,loss,learning_rate,epoch\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        logs = logs or {}
        step = int(state.global_step)

        loss = logs.get("loss", None)
        lr = logs.get("learning_rate", None)
        ep = logs.get("epoch", None)
        if (self._loss_csv is not None) and (loss is not None):
            self._loss_csv.write(f"{step},{loss},{lr},{ep}\n")
            self._loss_csv.flush()

        meta_list = getattr(self.trainer, "_last_meta", None) if self.trainer is not None else None
        if not meta_list:
            return

        picked: Dict[str, Dict[str, Any]] = {}
        for m in meta_list:
            lang = (m or {}).get("language", None)
            if (lang in self._fp) and (lang not in picked):
                picked[lang] = m

        for lang, m in picked.items():
            rec = {
                "step": step,
                "task": self.task,
                "language": lang,
                "dpo_setting": self.run_cfg.get("dpo_setting", {}),
                "logs": logs,
                "index": m.get("index", None),
                "prompt": m.get("prompt", None),
                "chosen": m.get("chosen", None),
                "rejected": m.get("rejected", None),
                "source_text": m.get("source_text", None),
                "hypothesis": m.get("hypothesis", None),
                "reward_total": m.get("reward_total", None),
                "reward_flip": m.get("reward_flip", None),
                "reward_edit": m.get("reward_edit", None),
                "reward_align": m.get("reward_align", None),
                "reward_aug": m.get("reward_aug", None),
            }
            self._fp[lang].write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._fp[lang].flush()

    def on_train_end(self, args, state, control, **kwargs):
        for fp in self._fp.values():
            fp.close()
        if self._loss_csv is not None:
            self._loss_csv.close()


def get_parser():
    ap = argparse.ArgumentParser("Multilingual DPO training (LoRA) with step logs + wandb + SFT warmup.")

    # Task / data IO
    ap.add_argument("--task", type=str, required=True, choices=["xnli", "sib200", "taxi1500"], help="One task per run.")

    ap.add_argument("--triple_root", type=str, required=True,
                    help="Root of triples, e.g. ../data_qwen_dpo_triple_1flip+1edit+1aug+1align")
    ap.add_argument("--languages", type=str, nargs="+", required=True, help="Languages to include in this run.")
    ap.add_argument("--max_per_lang", type=str, nargs="*", default=[],
                    help="Optional per-lang cap like en=2000 de=2000; if omitted, use all.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shuffle", action="store_true", help="Shuffle merged dataset.")

    # Model / tokenizer
    ap.add_argument("--model_name_or_path", type=str, required=True, help="Base model path/ID.")
    ap.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN_PATH", None))
    ap.add_argument("--cache_dir", type=str, default=None)

    # Output dirs
    ap.add_argument("--output_dir", type=str, required=True, help="HF Trainer output_dir (checkpoints etc).")
    ap.add_argument("--step_log_root", type=str, default=None,
                    help="Where to save step logs. Default: parent(output_dir)/step_logs_{run_name}.")
    ap.add_argument("--run_name", type=str, default=None)

    # Adapter output
    ap.add_argument("--adapter_root", type=str, default="/autodl-tmp/Multilingual CFE RL/adapter",
                    help="Root directory for saving LoRA adapters.")
    ap.add_argument("--adapter_name", type=str, default=None,
                    help="Adapter subfolder name under adapter_root/task/. If None, auto-generated.")

    # W&B
    ap.add_argument("--wandb_project", type=str, default="dpo-multilingual-cfe")
    ap.add_argument("--wandb_mode", type=str, default=os.environ.get("WANDB_MODE", "online"),
                    choices=["online", "offline", "disabled"])
    ap.add_argument("--report_to", type=str, default="wandb", choices=["wandb", "none"])

    # DPO hyperparams
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--loss_type", type=str, default="sigmoid")
    ap.add_argument("--learning_rate", type=float, default=5e-6)
    ap.add_argument("--num_train_epochs", type=float, default=1.0)
    ap.add_argument("--per_device_train_batch_size", type=int, default=7)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=7)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--max_prompt_length", type=int, default=512)
    ap.add_argument("--max_completion_length", type=int, default=None)

    # Logging / saving
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_steps", type=int, default=50)
    ap.add_argument("--save_total_limit", type=int, default=3)

    # Memory knobs (TRL DPOConfig)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--use_logits_to_keep", action="store_true")
    ap.add_argument("--padding_free", action="store_true")
    ap.add_argument("--precompute_ref_log_probs", action="store_true")

    # Round-robin by language
    ap.add_argument("--round_robin_by_language", action="store_true")
    ap.add_argument("--rr_shuffle_within_lang", action="store_true")
    ap.add_argument("--rr_cycle", action="store_true")

    # LoRA
    ap.add_argument("--lora_scope", type=str, default="all",
                    choices=["all", "attn", "mlp", "qv", "custom"])
    ap.add_argument("--lora_r", type=int, default=128)
    ap.add_argument("--lora_alpha", type=int, default=256)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"])
    ap.add_argument("--lora_target_modules", type=str, nargs="+", default=None)

    # Ref logps dataset cache
    ap.add_argument("--ref_logps_cache_dir", type=str, default=None)

    # SFT warmup (DEFAULT ON). BooleanOptionalAction gives --sft_first / --no-sft_first
    ap.add_argument("--sft_first", action=argparse.BooleanOptionalAction, default=False,
                    help="Run SFT warmup before DPO (default: enabled).")
    ap.add_argument("--sft_epochs", type=float, default=2.0)
    ap.add_argument("--sft_lr", type=float, default=None)
    ap.add_argument("--sft_only_flip", action=argparse.BooleanOptionalAction, default=True)

    return ap


def load_triples(triple_root: Path, task: str, languages: List[str], max_per_lang: Dict[str, int]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for lang in languages:
        p = triple_root / task / "train" / f"{lang}.jsonl"
        if not p.exists():
            raise FileNotFoundError(f"Missing triple file: {p}")
        cap = max_per_lang.get(lang, None)
        n = 0
        for r in read_jsonl(p):
            if cap is not None and n >= cap:
                break

            meta: Dict[str, Any] = {
                "task": task,
                "language": r.get("language", lang),
                "index": r.get("index", None),
                "prompt": r.get("prompt", None),
                "chosen": r.get("chosen", None),
                "rejected": r.get("rejected", None),
            }

            if task == "xnli":
                meta["source_text"] = r.get("premise", None)
                meta["hypothesis"] = r.get("hypothesis", None)
            else:
                meta["source_text"] = r.get("text", None)
                meta["hypothesis"] = None

            cm = r.get("chosen_meta", {}) or {}
            meta["reward_total"] = cm.get("reward_total", None)
            meta["reward_flip"] = cm.get("reward_flip_raw", None)
            meta["reward_edit"] = cm.get("reward_edit_raw", None)
            meta["reward_align"] = cm.get("reward_align_raw", None)

            if cm.get("reward_flip_raw", 0.0) == 1.0:
                meta["reward_aug"] = cm.get("reward_aug_flip_raw", None)
            else:
                meta["reward_aug"] = cm.get("reward_aug_noflip_raw", None)

            rows.append({
                "prompt": r["prompt"].rstrip() + "\n",
                "chosen": r["chosen"].strip(),
                "rejected": r["rejected"].strip(),
                "meta": meta,
            })
            n += 1
    return rows


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.run_name is None:
        args.run_name = f"dpo_{args.task}_{datetime.now().strftime('%y%m%d_%H%M%S')}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.step_log_root is None:
        args.step_log_root = str(output_dir.parent / f"step_logs_{args.run_name}")
    step_log_root = Path(args.step_log_root)
    step_log_root.mkdir(parents=True, exist_ok=True)

    with (output_dir / "args.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(vars(args), f, allow_unicode=True)

    if args.wandb_mode == "disabled" or args.report_to == "none":
        os.environ["WANDB_DISABLED"] = "true"
    else:
        os.environ["WANDB_MODE"] = args.wandb_mode

    if args.report_to == "wandb" and args.wandb_mode != "disabled":
        wandb.init(project=args.wandb_project, name=args.run_name, dir=str(output_dir))

    max_per_lang = parse_kv_list(args.max_per_lang) if args.max_per_lang else {}
    triple_root = Path(args.triple_root)

    rows = load_triples(triple_root, args.task, args.languages, max_per_lang)
    if args.shuffle:
        random.shuffle(rows)

    ds = Dataset.from_list(rows)

    cache_dir = Path(args.ref_logps_cache_dir) if args.ref_logps_cache_dir else None
    if cache_dir is not None and cache_dir.exists():
        print(f"[ref_logps_cache] loading from: {cache_dir}")
        ds = load_from_disk(str(cache_dir))
        args.precompute_ref_log_probs = False

    ds = ensure_dpo_text_columns(ds)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        token=args.hf_token,
        cache_dir=args.cache_dir,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    torch_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else None)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        token=args.hf_token,
        cache_dir=args.cache_dir,
        device_map="auto",
        torch_dtype=torch_dtype,
        revision="main",
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        
    # ---- TRL workaround: Gemma* 只做文本任务时，强制走 text tokenize_row ----
    
    
    orig_model_type = getattr(model.config, "model_type", None)
    if orig_model_type and str(orig_model_type).lower().startswith("gemma"):
        model.config.model_type = "gemma_text_only"
        print(f"[workaround] override model_type: {orig_model_type} -> {model.config.model_type}")




    target_modules = resolve_lora_target_modules(args.lora_scope, args.lora_target_modules)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias=args.lora_bias,
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)

    # ---- SFT warmup (DEFAULT ON) ----
    # TRL recent versions use SFTConfig(max_length=...), not max_seq_length. [web:23][web:29]
    if args.sft_first:
        sft_ds = build_sft_dataset_from_dpo(ds, only_flip=bool(args.sft_only_flip))
        sft_out = output_dir / "sft_warmup"
        sft_out.mkdir(parents=True, exist_ok=True)

        sft_cfg = SFTConfig(
            output_dir=str(sft_out),
            run_name=args.run_name + "_sft",
            num_train_epochs=args.sft_epochs,
            learning_rate=(args.sft_lr if args.sft_lr is not None else args.learning_rate),
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_ratio=args.warmup_ratio,
            weight_decay=args.weight_decay,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            save_total_limit=args.save_total_limit,
            max_length=args.max_length,         # <-- FIX HERE
            bf16=args.bf16 if args.bf16 else None,
            fp16=args.fp16,
            report_to=(["wandb"] if args.report_to == "wandb" else []),
            dataset_text_field="text",
        )

        sft_trainer = SFTTrainer(
            model=model,
            args=sft_cfg,
            train_dataset=sft_ds,
            processing_class=tokenizer,
        )
        sft_trainer.train()

        if sft_trainer.is_world_process_zero():
            (output_dir / "lora_sft").mkdir(parents=True, exist_ok=True)
            sft_trainer.model.save_pretrained(str(output_dir / "lora_sft"))
            tokenizer.save_pretrained(str(output_dir / "lora_sft"))

    # ---- DPO ----
    dpo_cfg = DPOConfig(
        output_dir=str(output_dir),
        run_name=args.run_name,
        beta=args.beta,
        loss_type=args.loss_type,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        max_length=args.max_length,
        #max_prompt_length=args.max_prompt_length,
        #max_completion_length=args.max_completion_length,
        bf16=args.bf16 if args.bf16 else None,
        fp16=args.fp16,
        report_to=(["wandb"] if args.report_to == "wandb" else []),
        remove_unused_columns=False,
        #use_logits_to_keep=args.use_logits_to_keep,
        #padding_free=args.padding_free,
        #precompute_ref_log_probs=args.precompute_ref_log_probs,
    )

    rr_batch_sampler = None
    if args.round_robin_by_language:
        rr_batch_sampler = LanguageRoundRobinBatchSampler(
            dataset=ds,
            languages=args.languages,
            batch_size=args.per_device_train_batch_size,
            seed=args.seed,
            shuffle_within_lang=bool(args.rr_shuffle_within_lang),
            drop_last=True,
            cycle=bool(args.rr_cycle),
        )
        


    trainer = MetaDPOTrainer(
        model=model,
        args=dpo_cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        ref_model=None,
        rr_batch_sampler=rr_batch_sampler,
    )

    base_collator = trainer.data_collator

    def wrapped_collator(examples: List[Dict[str, Any]]):
        batch = base_collator(examples)
        batch["meta"] = [ex["meta"] for ex in examples]
        return batch

    trainer.data_collator = wrapped_collator

    run_cfg = {
        "dpo_setting": {
            "beta": args.beta,
            "loss_type": args.loss_type,
            "max_length": args.max_length,
            "max_prompt_length": args.max_prompt_length,
            "precompute_ref_log_probs": bool(args.precompute_ref_log_probs),
            "padding_free": bool(args.padding_free),
            "use_logits_to_keep": bool(args.use_logits_to_keep),
            "grad_accum": args.gradient_accumulation_steps,
            "bsz": args.per_device_train_batch_size,
            "lr": args.learning_rate,
            "round_robin_by_language": bool(args.round_robin_by_language),
            "rr_shuffle_within_lang": bool(args.rr_shuffle_within_lang),
            "rr_cycle": bool(args.rr_cycle),
            "lora_scope": args.lora_scope,
            "lora_target_modules": target_modules,
            "adapter_root": args.adapter_root,
            "adapter_name": args.adapter_name,
            "sft_first": bool(args.sft_first),
            "sft_epochs": args.sft_epochs,
            "sft_only_flip": bool(args.sft_only_flip),
            "sft_lr": (args.sft_lr if args.sft_lr is not None else args.learning_rate),
        }
    }
    cb = StepFileLoggerCallback(step_log_root=step_log_root, task=args.task, languages=args.languages, run_cfg=run_cfg)
    cb.trainer = trainer
    trainer.add_callback(cb)

    if cache_dir is not None and (not cache_dir.exists()) and args.precompute_ref_log_probs:
        print(f"[ref_logps_cache] precomputing & saving to: {cache_dir}")
        _ = trainer.get_train_dataloader()
        if trainer.is_world_process_zero():
            trainer.train_dataset.save_to_disk(str(cache_dir))

    trainer.train()

    adapter_root = Path(args.adapter_root)
    adapter_root.mkdir(parents=True, exist_ok=True)

    if args.adapter_name is None:
        ts = datetime.now().strftime("%y%m%d_%H%M%S")
        args.adapter_name = f"{args.run_name}__{ts}"

    adapter_dir = adapter_root / args.task / args.adapter_name
    if trainer.is_world_process_zero():
        adapter_dir.mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        print(f"[save] LoRA adapter saved to: {adapter_dir}")

        (output_dir / "lora").mkdir(parents=True, exist_ok=True)
        trainer.model.save_pretrained(str(output_dir / "lora"))
        tokenizer.save_pretrained(str(output_dir / "lora"))


if __name__ == "__main__":
    main(get_parser().parse_args())
