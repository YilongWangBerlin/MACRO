#!/usr/bin/env python3
import argparse
import datetime as _datetime
import json
import os
import random
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import yaml
import wandb
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback

from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig


datetime = _datetime.datetime


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def parse_kv_list(items: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"Bad kv item: {it}, expected like en=1000")
        k, v = it.split("=", 1)
        out[k.strip()] = int(v.strip())
    return out


def resolve_lora_target_modules(scope: str, custom: Optional[List[str]]) -> List[str]:
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


def load_sft_rows(
    triple_root: Path,
    task: str,
    languages: List[str],
    max_per_lang: Dict[str, int],
) -> List[Dict[str, Any]]:
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

            prompt = (r.get("prompt") or "").rstrip()
            chosen = (r.get("chosen") or "").strip()

            if not chosen:
                continue

            text = f"{prompt}\n{chosen}" if prompt else chosen

            rows.append({
                "text": text,
                "language": r.get("language", lang),
                "index": r.get("index", None),
            })
            n += 1

    return rows


class TokenizerProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)


class TrainLossLoggerCallback(TrainerCallback):
    def __init__(self, log_root: Path, task: str):
        self.log_root = log_root
        self.task = task
        self._loss_csv = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.log_root.mkdir(parents=True, exist_ok=True)
        loss_csv = self.log_root / f"{self.task}_loss.csv"
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

        if self._loss_csv is not None and loss is not None:
            self._loss_csv.write(f"{step},{loss},{lr},{ep}\n")
            self._loss_csv.flush()

    def on_train_end(self, args, state, control, **kwargs):
        if self._loss_csv is not None:
            self._loss_csv.close()


def get_parser():
    ap = argparse.ArgumentParser("Multilingual SFT training (LoRA, chosen-only).")

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
    ap.add_argument("--log_root", type=str, default=None,
                    help="Where to save loss csv. Default: parent(output_dir)/train_logs_{run_name}.")
    ap.add_argument("--run_name", type=str, default=None)

    # Adapter output
    ap.add_argument("--adapter_root", type=str, default="/autodl-tmp/Multilingual CFE RL/adapter",
                    help="Root directory for saving LoRA adapters.")
    ap.add_argument("--adapter_name", type=str, default=None,
                    help="Adapter subfolder name under adapter_root/task/. If None, auto-generated.")

    # W&B
    ap.add_argument("--wandb_project", type=str, default="sft-multilingual-cfe")
    ap.add_argument("--wandb_mode", type=str, default=os.environ.get("WANDB_MODE", "online"),
                    choices=["online", "offline", "disabled"])
    ap.add_argument("--report_to", type=str, default="wandb", choices=["wandb", "none"])

    # SFT hyperparams
    ap.add_argument("--learning_rate", type=float, default=5e-6)
    ap.add_argument("--num_train_epochs", type=float, default=2.0)
    ap.add_argument("--per_device_train_batch_size", type=int, default=7)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=7)
    ap.add_argument("--warmup_ratio", type=float, default=0.03)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_length", type=int, default=1024)

    # Logging / saving
    ap.add_argument("--logging_steps", type=int, default=10)
    ap.add_argument("--save_steps", type=int, default=50)
    ap.add_argument("--save_total_limit", type=int, default=3)

    # Precision / memory
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--gradient_checkpointing", action="store_true")

    # LoRA
    ap.add_argument("--lora_scope", type=str, default="all",
                    choices=["all", "attn", "mlp", "qv", "custom"])
    ap.add_argument("--lora_r", type=int, default=128)
    ap.add_argument("--lora_alpha", type=int, default=256)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"])
    ap.add_argument("--lora_target_modules", type=str, nargs="+", default=None)

    return ap


def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.run_name is None:
        args.run_name = f"sft_{args.task}_{datetime.now().strftime('%y%m%d_%H%M%S')}"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.log_root is None:
        args.log_root = str(output_dir.parent / f"train_logs_{args.run_name}")
    log_root = Path(args.log_root)
    log_root.mkdir(parents=True, exist_ok=True)

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

    rows = load_sft_rows(triple_root, args.task, args.languages, max_per_lang)
    if args.shuffle:
        random.shuffle(rows)

    if len(rows) == 0:
        raise ValueError("No valid chosen samples loaded.")

    ds = Dataset.from_list(rows)

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

    sft_cfg = SFTConfig(
        output_dir=str(output_dir),
        run_name=args.run_name,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        max_length=args.max_length,
        bf16=args.bf16 if args.bf16 else None,
        fp16=args.fp16,
        report_to=(["wandb"] if args.report_to == "wandb" else []),
        dataset_text_field="text",
    )

    processor = TokenizerProcessor(tokenizer)

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=ds,
        processing_class=tokenizer,
    )

    trainer.add_callback(TrainLossLoggerCallback(log_root=log_root, task=args.task))
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

    if args.report_to == "wandb" and args.wandb_mode != "disabled":
        wandb.finish()


if __name__ == "__main__":
    main(get_parser().parse_args())