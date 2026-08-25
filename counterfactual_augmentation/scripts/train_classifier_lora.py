"""
LoRA fine-tunes a base model for text classification (SIB200/TAXI1500) on a
prepared {prompt, completion} jsonl. Loss is computed only
on the completion tokens (prompt tokens masked with -100), matching standard
instruction-completion SFT practice. Saves the LoRA adapter to --output_dir
(a scratch/checkpoint directory, matching where the paper's own adapters live).

LoRA hyperparameters (r=128, alpha=256, dropout=0.05, target_modules=all
attn+mlp projections) mirror train/sft.py's defaults for the paper's own SFT
baseline, for methodological consistency.
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments,
    DataCollatorForSeq2Seq, EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model


def preprocess_logits_for_metrics(logits, labels):
    """Reduce full-vocab logits to argmax token ids before accumulation, so eval
    doesn't have to hold [batch, seq_len, vocab_size] tensors in memory."""
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_pred):
    """Per-example completion accuracy under teacher forcing: an example counts as
    correct only if EVERY completion token (labels != -100) is predicted correctly.
    Since completions are single-word labels, this closely approximates the
    greedy-decoded classification accuracy used at eval time."""
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
    # next-token prediction: logits/preds at position t predict input_ids[t+1]
    preds = preds[:, :-1]
    labels = labels[:, 1:]
    n_correct, n_total = 0, 0
    for p_row, l_row in zip(preds, labels):
        mask = l_row != -100
        if not mask.any():
            continue
        n_total += 1
        if np.array_equal(p_row[mask], l_row[mask]):
            n_correct += 1
    return {"accuracy": (n_correct / n_total) if n_total else 0.0}


class PromptCompletionDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length):
        self.examples = []
        for r in rows:
            prompt_ids = tokenizer(r["prompt"], add_special_tokens=True)["input_ids"]
            full_text = r["prompt"] + r["completion"]
            full_ids = tokenizer(full_text, add_special_tokens=True)["input_ids"]
            if tokenizer.eos_token_id is not None and full_ids[-1] != tokenizer.eos_token_id:
                full_ids = full_ids + [tokenizer.eos_token_id]
            full_ids = full_ids[:max_length]
            labels = list(full_ids)
            plen = min(len(prompt_ids), len(full_ids))
            for i in range(plen):
                labels[i] = -100
            self.examples.append({"input_ids": full_ids, "labels": labels,
                                   "attention_mask": [1] * len(full_ids)})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_file", type=str, required=True)
    ap.add_argument("--validation_file", type=str, default=None,
                     help="optional {prompt,completion} jsonl for eval_loss monitoring during training")
    ap.add_argument("--eval_steps", type=int, default=100)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=8)
    ap.add_argument("--early_stopping_patience", type=int, default=3,
                     help="stop once the early-stopping metric fails to improve for this many "
                          "consecutive evals; only active when --validation_file is given. The best "
                          "checkpoint (by that metric) is what gets saved, not the last one.")
    ap.add_argument("--early_stopping_metric", type=str, default="loss", choices=["loss", "accuracy"],
                     help="'loss': lowest eval_loss (LM cross-entropy). 'accuracy': highest per-example "
                          "teacher-forced completion accuracy on the validation set (a proxy for "
                          "classification accuracy, cheaper than running generation every eval step).")
    ap.add_argument("--base_model", type=str, required=True)
    ap.add_argument("--init_adapter_path", type=str, default=None,
                     help="if set, continue training from this existing saved LoRA adapter instead of "
                          "initializing a fresh one (--lora_r/--lora_alpha/--lora_dropout are ignored, "
                          "the adapter's own config is used). Useful for extending a completed run by "
                          "more epochs without retraining from scratch.")
    ap.add_argument("--output_dir", type=str, required=True, help="where to save the LoRA adapter")
    ap.add_argument("--num_train_epochs", type=float, default=2.0)
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--per_device_train_batch_size", type=int, default=8,
                     help="with --auto_find_batch_size, this is the STARTING ceiling to try, not a fixed "
                          "value: accelerate's find_executable_batch_size halves it on CUDA OOM until "
                          "training fits, so pass a deliberately high number (e.g. 32-64) to let it "
                          "actually search down to whatever the allocated GPU can hold, instead of "
                          "hand-picking a conservative constant per model as the old per-model launcher "
                          "scripts did.")
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--auto_find_batch_size", action="store_true",
                     help="auto-adapt per_device_train_batch_size to the allocated GPU's free memory "
                          "(halving from --per_device_train_batch_size on CUDA OOM via accelerate's "
                          "find_executable_batch_size) instead of using a fixed value. Recommended for "
                          "job arrays that can land on GPUs of very different memory sizes.")
    ap.add_argument("--max_length", type=int, default=320)
    ap.add_argument("--lora_r", type=int, default=128)
    ap.add_argument("--lora_alpha", type=int, default=256)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = [json.loads(l) for l in Path(args.train_file).read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"[info] loaded {len(rows)} training rows from {args.train_file}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, device_map="balanced", trust_remote_code=True, torch_dtype=torch.bfloat16,
    )
    if args.init_adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.init_adapter_path, is_trainable=True)
        print(f"[info] continuing training from existing adapter at {args.init_adapter_path}", flush=True)
    else:
        lora_cfg = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    model.config.use_cache = False

    dataset = PromptCompletionDataset(rows, tokenizer, args.max_length)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, label_pad_token_id=-100)

    eval_dataset = None
    if args.validation_file:
        val_rows = [json.loads(l) for l in Path(args.validation_file).read_text(encoding="utf-8").splitlines() if l.strip()]
        eval_dataset = PromptCompletionDataset(val_rows, tokenizer, args.max_length)
        print(f"[info] loaded {len(val_rows)} validation rows from {args.validation_file}", flush=True)

    targs = TrainingArguments(
        output_dir=str(Path(args.output_dir) / "_trainer_ckpts"),
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_ratio=0.03,
        weight_decay=0.0,
        bf16=True,
        logging_steps=20,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        save_strategy="steps" if eval_dataset is not None else "no",
        save_steps=args.eval_steps if eval_dataset is not None else None,
        save_total_limit=3,
        # NOTE: load_best_model_at_end=True is intentionally NOT used here. With this
        # transformers/peft combo, Trainer's end-of-training reload calls
        # PeftModel.load_adapter() -> set_peft_model_state_dict() ->
        # _maybe_shard_state_dict_for_tp(), which does an unconditional
        # `from transformers.integrations.tensor_parallel import EmbeddingParallel`
        # at the top of the function -- ImportError on transformers==4.56.2 (that
        # symbol doesn't exist there), regardless of whether TP is actually used.
        # This crashes AFTER training has fully completed, losing the run. We track
        # the best checkpoint via `metric_for_best_model` bookkeeping (still populated
        # without the reload) and load it ourselves below with a raw state-dict load,
        # bypassing PeftModel.load_adapter() entirely.
        load_best_model_at_end=False,
        metric_for_best_model="accuracy" if args.early_stopping_metric == "accuracy" else "loss",
        greater_is_better=(args.early_stopping_metric == "accuracy"),
        report_to=[],
        gradient_checkpointing=False,
        remove_unused_columns=False,
        auto_find_batch_size=args.auto_find_batch_size,
    )

    callbacks = []
    if eval_dataset is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    use_acc = eval_dataset is not None and args.early_stopping_metric == "accuracy"
    trainer = Trainer(
        model=model, args=targs, train_dataset=dataset, eval_dataset=eval_dataset,
        data_collator=collator, callbacks=callbacks,
        compute_metrics=compute_metrics if use_acc else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if use_acc else None,
    )
    trainer.train()
    if args.auto_find_batch_size:
        used_bs = getattr(trainer, "_train_batch_size", args.per_device_train_batch_size)
        print(f"[info] auto_find_batch_size: trained with per_device_train_batch_size={used_bs} "
              f"(ceiling was {args.per_device_train_batch_size})", flush=True)
    if eval_dataset is not None:
        best = trainer.state.best_metric
        best_ckpt = trainer.state.best_model_checkpoint
        print(f"[info] training stopped; best {args.early_stopping_metric}={best} at {best_ckpt}", flush=True)
        if best_ckpt is not None:
            from safetensors.torch import load_file
            sd_path = Path(best_ckpt) / "adapter_model.safetensors"
            state_dict = load_file(str(sd_path))
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            unexpected = [k for k in unexpected]  # non-LoRA base-model keys are expected to be "missing"
            print(f"[info] manually loaded best checkpoint weights from {sd_path} "
                  f"(unexpected_keys={len(unexpected)})", flush=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"[ok] saved LoRA adapter (best {args.early_stopping_metric} checkpoint if validation was used) -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
