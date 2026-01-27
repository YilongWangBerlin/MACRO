#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Add `target_pred` to jsonl files under ../data_qwen_pred.

NEW:
- If orig_pred is invalid (None / non-int / out of range), re-predict it locally
  using the same prompt & generation setting you provided (Gemma/Qwen style).
- Then sample target_pred with constraints:
  target_pred != label and target_pred != orig_pred
- For non-English languages, prefer matching en target_pred for same index;
  if conflict, sample a valid target randomly.

This script processes: {xnli,sib200}/{train,validation,test}/*.jsonl
and overwrites files in-place (via atomic replace).
"""

import os
import re
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Any

# ---------- optional: only needed when repair happens ----------
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ----------------------------
# Configuration
# ----------------------------

DATA_ROOT = Path("../data_qwen_pred")
DATASETS = ["xnli", "sib200"]
SPLITS = ["train", "validation", "test"]
LANGS = ["en", "ar", "de", "ru", "sw", "vi", "zh"]

NUM_LABELS = {"xnli": 3, "sib200": 7}
RANDOM_SEED = 569

REPAIR_ENABLED = True
REPAIR_MODEL_DIR = "/root/autodl-tmp/model/gemma-2-9b-it"
REPAIR_DEVICE_MAP = "auto"
REPAIR_DTYPE = "bfloat16"  # "float16"/"bfloat16"/"float32"
REPAIR_USE_CHAT_TEMPLATE = True
REPAIR_MAX_NEW_TOKENS = 1
REPAIR_TEMPERATURE = 0.0
REPAIR_MAX_RETRIES = 3


SIB200_LABEL2ID = {
    "science/technology": 0,
    "travel": 1,
    "politics": 2,
    "sports": 3,
    "health": 4,
    "entertainment": 5,
    "geography": 6,
}

_INT_RE = re.compile(r"(-?\d+)")

# ----------------------------
# IO Helpers
# ----------------------------

def read_jsonl(path: Path) -> List[dict]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSON parse error in {path} at line {line_no}: {e}") from e
    return items


def write_jsonl_atomic(path: Path, items: List[dict]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)

# ----------------------------
# Pred parsing / validation
# ----------------------------

def parse_first_int(s: str) -> Optional[int]:
    if s is None:
        return None
    m = _INT_RE.search(str(s).strip())
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def is_valid_pred(v: Any, num_labels: int) -> bool:
    return isinstance(v, int) and 0 <= v < num_labels


def get_pred_from_fields(obj: dict, lang: str, num_labels: int) -> Optional[int]:
    # 1) orig_pred[lang]
    d = obj.get("orig_pred", {})
    if isinstance(d, dict) and lang in d:
        v = d[lang]
        if isinstance(v, int) and is_valid_pred(v, num_labels):
            return v
        if isinstance(v, str) and v.strip().isdigit():
            vv = int(v.strip())
            if is_valid_pred(vv, num_labels):
                return vv
        # v could be None -> fallthrough

    # 2) try parse from orig_pred_text[lang]
    dt = obj.get("orig_pred_text", {})
    if isinstance(dt, dict) and lang in dt:
        vv = parse_first_int(dt.get(lang))
        if is_valid_pred(vv, num_labels):
            return vv

    return None

# ----------------------------
# Repair predictor (lazy)
# ----------------------------

def _torch_dtype(name: str):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


class RepairPredictor:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.tokenizer = None
        self.model = None

    def ensure_loaded(self):
        if self.model is not None and self.tokenizer is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_dir,
            trust_remote_code=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            device_map=REPAIR_DEVICE_MAP,
            torch_dtype=_torch_dtype(REPAIR_DTYPE),
            trust_remote_code=True,
        )
        # pad safety
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def encode_prompt(self, prompt_text: str):
        if REPAIR_USE_CHAT_TEMPLATE and hasattr(self.tokenizer, "apply_chat_template"):
            # 注意：Gemma-*-it 不支持 system role，这里只用 user role。[web:1]
            messages = [{"role": "user", "content": prompt_text}]
            enc = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                return_dict=True,
                add_generation_prompt=True,  # 让模板补上“开始生成”的标记。[web:106]
            )
            return enc
        else:
            return self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)

    def generate_text(self, prompt_text: str) -> str:
        self.ensure_loaded()
        enc = self.encode_prompt(prompt_text)
        input_ids = enc["input_ids"]
        attn = enc.get("attention_mask", None)

        # move to model device (works for single-GPU; device_map=auto handled by HF internally for generate)
        if hasattr(self.model, "device") and str(self.model.device) != "meta":
            input_ids = input_ids.to(self.model.device)
            if attn is not None:
                attn = attn.to(self.model.device)

        prompt_len = input_ids.shape[1]

        with torch.inference_mode():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attn,
                max_new_tokens=REPAIR_MAX_NEW_TOKENS,
                do_sample=False if REPAIR_TEMPERATURE <= 0 else True,
                temperature=None if REPAIR_TEMPERATURE <= 0 else REPAIR_TEMPERATURE,
                top_p=1.0,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        gen_ids = out[0][prompt_len:]
        return self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

# ----------------------------
# Prompts (same as you provided)
# ----------------------------

def build_prompt_xnli(lang: str, premise: str, hypothesis: str) -> str:
    system = (
        "You are a multilingual natural language inference (NLI) classifier./no_think\n"
        "Task: Given a Premise and a Hypothesis in the SAME language, output their relation label.\n"
        "Labels (output exactly ONE digit):\n"
        "0 = entailment (Premise makes Hypothesis definitely true)\n"
        "1 = neutral (not enough info; could be true or false)\n"
        "2 = contradiction (Premise makes Hypothesis definitely false)\n"
        "Rules:\n"
        "- Output MUST be exactly one digit (0/1/2).\n"
        "- Do NOT output words, punctuation, spaces, or newlines."
    )
    user = (
        f"Language: {lang}\n"
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        "Label: "
    )
    return system + "\n\n" + user


def build_prompt_sib200(lang: str, text: str) -> str:
    mapping_lines = "\n".join([f"{k} -> {v}" for k, v in SIB200_LABEL2ID.items()])
    system = (
        "You are a multilingual topic classifier for short news sentences./no_think\n"
        "Assign exactly one topic label ID (0-6) to the given news text.\n"
        "Label mapping:\n"
        f"{mapping_lines}\n"
        "Guidelines:\n"
        "- Choose the single best topic that the text is mainly about.\n"
        "- If multiple topics appear, pick the primary event/subject.\n"
        "- If unsure, pick the closest single topic.\n\n"
        "Output rules:\n"
        "- Output EXACTLY ONE character from {0,1,2,3,4,5,6}.\n"
        "- Do NOT output words, punctuation, spaces, or newlines."
    )
    user = (
        f"Language: {lang}\n"
        f"Text: {text}\n"
        "Label: "
    )
    return system + "\n\n" + user


def repredict_if_needed(
    obj: dict,
    dataset: str,
    lang: str,
    predictor: RepairPredictor,
) -> int:
    num_labels = NUM_LABELS[dataset]
    pred = get_pred_from_fields(obj, lang, num_labels=num_labels)
    if pred is not None:
        # write back normalized int
        obj.setdefault("orig_pred", {})[lang] = int(pred)
        return int(pred)

    if not REPAIR_ENABLED:
        raise RuntimeError(f"Invalid orig_pred and REPAIR_DISABLED, index={obj.get('index')} lang={lang}")

    # build prompt from existing fields in pred jsonl
    if dataset == "xnli":
        premise = obj.get("premise", {}).get(lang)
        hypothesis = obj.get("hypothesis", {}).get(lang)
        if premise is None or hypothesis is None:
            raise RuntimeError(f"Missing premise/hypothesis for repair, index={obj.get('index')} lang={lang}")
        prompt = build_prompt_xnli(lang, premise, hypothesis)
    else:
        text = obj.get("text", {}).get(lang)
        if text is None:
            raise RuntimeError(f"Missing text for repair, index={obj.get('index')} lang={lang}")
        prompt = build_prompt_sib200(lang, text)

    last_text = None
    for _ in range(REPAIR_MAX_RETRIES):
        last_text = predictor.generate_text(prompt)
        pred2 = parse_first_int(last_text)
        if is_valid_pred(pred2, num_labels):
            obj.setdefault("orig_pred", {})[lang] = int(pred2)
            obj.setdefault("orig_pred_text", {})[lang] = last_text
            return int(pred2)

    # fallback: random valid pred (different from label if possible)
    label = obj.get("label", None)
    candidates = list(range(num_labels))
    if isinstance(label, int) and 0 <= label < num_labels and num_labels > 1:
        candidates = [c for c in candidates if c != label] or list(range(num_labels))
    pred3 = random.choice(candidates)

    obj.setdefault("orig_pred", {})[lang] = int(pred3)
    obj.setdefault("orig_pred_text", {})[lang] = last_text if last_text is not None else ""
    obj.setdefault("repair_warning", {})[lang] = "repredict_failed_fallback_random"
    return int(pred3)

# ----------------------------
# Target sampling
# ----------------------------

def choose_target(num_labels: int, label: int, orig_pred: int, preferred: Optional[int] = None) -> int:
    excluded = {label, orig_pred}
    candidates = [c for c in range(num_labels) if c not in excluded]
    if not candidates:
        raise RuntimeError(f"No candidates left after excluding label={label}, orig_pred={orig_pred}")
    if preferred is not None and preferred in candidates:
        return preferred
    return random.choice(candidates)


def build_en_target_map(items: List[dict], dataset: str, predictor: RepairPredictor) -> Dict[int, int]:
    num_labels = NUM_LABELS[dataset]
    mp: Dict[int, int] = {}
    for obj in items:
        idx = obj.get("index", None)
        if idx is None:
            raise RuntimeError("Missing 'index' field in an item.")
        label = obj.get("label", None)
        if label is None:
            raise RuntimeError(f"Missing 'label' for index={idx}")

        orig_pred = repredict_if_needed(obj, dataset=dataset, lang="en", predictor=predictor)

        en_target = choose_target(num_labels=num_labels, label=label, orig_pred=orig_pred, preferred=None)
        mp[idx] = en_target
        obj["target_pred"] = {"en": en_target}
    return mp


def apply_targets_for_lang(items: List[dict], dataset: str, lang: str, en_target_map: Dict[int, int], predictor: RepairPredictor) -> None:
    num_labels = NUM_LABELS[dataset]
    for obj in items:
        idx = obj.get("index", None)
        if idx is None:
            raise RuntimeError("Missing 'index' field in an item.")
        label = obj.get("label", None)
        if label is None:
            raise RuntimeError(f"Missing 'label' for index={idx}")

        orig_pred = repredict_if_needed(obj, dataset=dataset, lang=lang, predictor=predictor)

        preferred = en_target_map.get(idx, None)
        tgt = choose_target(num_labels=num_labels, label=label, orig_pred=orig_pred, preferred=preferred)
        obj["target_pred"] = {lang: tgt}


def process_split(dataset: str, split: str, predictor: RepairPredictor) -> None:
    split_dir = DATA_ROOT / dataset / split
    if not split_dir.exists():
        print(f"[skip] Missing directory: {split_dir}")
        return

    en_path = split_dir / "en.jsonl"
    if not en_path.exists():
        print(f"[skip] Missing en.jsonl: {en_path}")
        return

    en_items = read_jsonl(en_path)
    en_target_map = build_en_target_map(en_items, dataset=dataset, predictor=predictor)
    write_jsonl_atomic(en_path, en_items)
    print(f"[ok] Wrote targets: {en_path}")

    for lang in LANGS:
        if lang == "en":
            continue
        path = split_dir / f"{lang}.jsonl"
        if not path.exists():
            print(f"[skip] Missing file: {path}")
            continue

        items = read_jsonl(path)
        apply_targets_for_lang(items, dataset=dataset, lang=lang, en_target_map=en_target_map, predictor=predictor)
        write_jsonl_atomic(path, items)
        print(f"[ok] Wrote targets: {path}")


def main() -> None:
    random.seed(RANDOM_SEED)

    if not DATA_ROOT.exists():
        raise RuntimeError(f"DATA_ROOT does not exist: {DATA_ROOT.resolve()}")

    predictor = RepairPredictor(REPAIR_MODEL_DIR)

    for dataset in DATASETS:
        if dataset not in NUM_LABELS:
            raise RuntimeError(f"NUM_LABELS not configured for dataset={dataset}")
        for split in SPLITS:
            process_split(dataset=dataset, split=split, predictor=predictor)


if __name__ == "__main__":
    main()
