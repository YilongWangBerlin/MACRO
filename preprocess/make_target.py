#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Add `target_pred` to jsonl files under data_root.

- If orig_pred is invalid (None / non-int / out of range), re-predict it locally
  using constrained-label decoding (label words, not digits).
- Then sample target_pred with constraints:
  target_pred != label and target_pred != orig_pred
- For non-English languages, prefer matching en target_pred for same index;
  if conflict, sample a valid target randomly.

Processes: {xnli,sib200}/{train,validation,test}/*.jsonl
Overwrites files in-place (atomic replace).
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ----------------------------
# Label specs
# ----------------------------
NUM_LABELS = {"xnli": 3, "sib200": 7, "taxi1500": 6}


SIB200_LABEL2ID = {
    "science/technology": 0,
    "travel": 1,
    "politics": 2,
    "sports": 3,
    "health": 4,
    "entertainment": 5,
    "geography": 6,
}
SIB200_ID2LABEL = {v: k for k, v in SIB200_LABEL2ID.items()}

XNLI_ID2LABEL = {0: "entailment", 1: "neutral", 2: "contradiction"}
XNLI_LABEL2ID = {v: k for k, v in XNLI_ID2LABEL.items()}

TAXI1500_LABEL2ID = {
    "recommendation": 0,
    "faith": 1,
    "violence": 2,
    "grace": 3,
    "sin": 4,
    "description": 5,
}
TAXI1500_ID2LABEL = {v: k for k, v in TAXI1500_LABEL2ID.items()}

LANG_CODE2NAME = {
    "en": "English",
    "ar": "Arabic",
    "de": "German",
    "ru": "Russian",
    "sw": "Swahili",
    "vi": "Vietnamese",
    "zh": "Chinese",
}


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
# Pred validation / extraction
# ----------------------------
def is_valid_pred(v: Any, num_labels: int) -> bool:
    return isinstance(v, int) and 0 <= v < num_labels


def get_pred_from_fields(obj: dict, lang: str, num_labels: int) -> Optional[int]:
    d = obj.get("orig_pred", {})
    if isinstance(d, dict) and lang in d:
        v = d[lang]
        if isinstance(v, int) and is_valid_pred(v, num_labels):
            return v
        if isinstance(v, str) and v.strip().isdigit():
            vv = int(v.strip())
            if is_valid_pred(vv, num_labels):
                return vv
    return None


# ----------------------------
# Generation utils
# ----------------------------
def _torch_dtype(name: str):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def _normalize_eos_id(tokenizer) -> int:
    eos_id = tokenizer.eos_token_id
    if isinstance(eos_id, (list, tuple)):
        if len(eos_id) == 0:
            raise RuntimeError("tokenizer.eos_token_id is empty list/tuple")
        eos_id = eos_id[0]
    if not isinstance(eos_id, int):
        raise RuntimeError(f"tokenizer.eos_token_id is not int after normalize, got {type(eos_id)}")
    return eos_id


def encode_prompt(tokenizer, prompt_text: str, use_chat_template: bool = True) -> Dict[str, torch.Tensor]:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt_text}]
        enc = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
    else:
        enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)

    if isinstance(enc, torch.Tensor):
        return {"input_ids": enc}
    if hasattr(enc, "input_ids") and isinstance(enc.input_ids, torch.Tensor):
        out = {"input_ids": enc.input_ids}
        if hasattr(enc, "attention_mask") and isinstance(enc.attention_mask, torch.Tensor):
            out["attention_mask"] = enc.attention_mask
        return out
    if isinstance(enc, dict):
        return enc
    raise TypeError(f"Unexpected encode output type: {type(enc)}")


def normalize_label_text(s: str) -> str:
    return (s or "").strip().lower()


# ----------------------------
# Constrained decoding trie
# ----------------------------
class LabelTrie:
    def __init__(self):
        self.children: Dict[int, "LabelTrie"] = {}
        self.is_end: bool = False

    def insert(self, token_seq: List[int]) -> None:
        node = self
        for t in token_seq:
            if t not in node.children:
                node.children[t] = LabelTrie()
            node = node.children[t]
        node.is_end = True

    def next_tokens(self, prefix: List[int]) -> Tuple[bool, List[int]]:
        node = self
        for t in prefix:
            if t not in node.children:
                return (False, [])
            node = node.children[t]
        return (node.is_end, list(node.children.keys()))


def _unique_token_seqs(tokenizer, label: str) -> List[List[int]]:
    variants = [label, " " + label, "\n" + label]
    out = []
    seen = set()
    for v in variants:
        ids = tokenizer.encode(v, add_special_tokens=False)
        if not ids:
            continue
        key = tuple(ids)
        if key not in seen:
            seen.add(key)
            out.append(ids)
    return out


def build_label_trie(tokenizer, labels: List[str]) -> Tuple[LabelTrie, int]:
    trie = LabelTrie()
    max_len = 0
    for lab in labels:
        for seq in _unique_token_seqs(tokenizer, lab):
            trie.insert(seq)
            max_len = max(max_len, len(seq))
    return trie, max_len


def make_prefix_allowed_tokens_fn(trie: LabelTrie, prompt_len: int, eos_token_id: int):
    # Compatible with HF versions that call fn(batch_id, sent) where sent is 1D.
    def _fn(batch_id: int, input_ids: torch.Tensor) -> List[int]:
        if isinstance(input_ids, torch.Tensor):
            if input_ids.ndim == 2:
                seq = input_ids[batch_id].tolist()
            elif input_ids.ndim == 1:
                seq = input_ids.tolist()
            elif input_ids.ndim == 0:
                seq = [int(input_ids.item())]
            else:
                seq = input_ids.reshape(-1).tolist()
        else:
            if isinstance(input_ids, int):
                seq = [input_ids]
            else:
                seq = list(input_ids)

        cut = prompt_len if prompt_len <= len(seq) else len(seq)
        gen_prefix = seq[cut:]

        is_end, nxt = trie.next_tokens(gen_prefix)
        if nxt:
            return nxt
        if is_end:
            return [eos_token_id]
        return [eos_token_id]
    return _fn


# ----------------------------
# Candidate token-level logits
# ----------------------------
def score_label_candidates_token_logits(model, tokenizer, prompt_input_ids, prompt_attention_mask, candidate_labels):
    device = prompt_input_ids.device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else _normalize_eos_id(tokenizer)

    prompt_ids = prompt_input_ids[0].tolist()
    prompt_len = len(prompt_ids)

    cand_token_ids = []
    for lab in candidate_labels:
        ids = tokenizer.encode(lab, add_special_tokens=False)
        if not ids:
            ids = tokenizer.encode(" " + lab, add_special_tokens=False)
        cand_token_ids.append(ids)

    max_total = max(prompt_len + len(ids) for ids in cand_token_ids)
    bs = len(candidate_labels)

    input_ids = torch.full((bs, max_total), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((bs, max_total), dtype=torch.long, device=device)

    for i, ids in enumerate(cand_token_ids):
        seq = prompt_ids + ids
        input_ids[i, :len(seq)] = torch.tensor(seq, dtype=torch.long, device=device)
        attn[i, :len(seq)] = 1

    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits  # [bs, T, vocab]

    result = {}
    for i, lab in enumerate(candidate_labels):
        ids = cand_token_ids[i]
        tok_logits = []
        for j, tok_id in enumerate(ids):
            pos = prompt_len + j
            tok_logits.append(float(logits[i, pos - 1, tok_id].item()))
        result[lab] = {
            "token_ids": ids,
            "token_logits": tok_logits,
            "sum_token_logits": float(sum(tok_logits)),
        }
    return result


# ----------------------------
# Prompts
# ----------------------------
def build_prompt_xnli(lang: str, premise: str, hypothesis: str) -> str:
    lang_name = LANG_CODE2NAME.get(lang, lang)
    system = (
        "You are a multilingual Natural Language Inference (NLI) classifier.\n"
        "You must output exactly ONE label word from this closed set:\n"
        "entailment\nneutral\ncontradiction\n\n"
        "Output rules:\n"
        "- Output exactly one of the label words above.\n"
        "- No extra words, punctuation, quotes, spaces, or newlines.\n\n"
        "Example:\n"
        "Language: English\n"
        "Premise: A dog is running in the park.\n"
        "Hypothesis: An animal is running outdoors.\n"
        "Label: entailment\n"
    )
    user = (
        f"Language: {lang_name}\n"
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        "Label:"
    )
    return system + "\n\n" + user


def build_prompt_sib200(lang: str, text: str) -> str:
    lang_name = LANG_CODE2NAME.get(lang, lang)
    labels = list(SIB200_LABEL2ID.keys())
    label_block = "\n".join(labels)
    system = (
        "You are a multilingual topic classifier for short news sentences.\n"
        "You must output exactly ONE label word from this closed set:\n"
        f"{label_block}\n\n"
        "Output rules:\n"
        "- Output exactly one label from the set above.\n"
        "- No extra words, punctuation, quotes, spaces, or newlines.\n\n"
        "Example:\n"
        "Language: English\n"
        "Text: The prime minister announced new election dates.\n"
        "Label: politics\n"
    )
    user = (
        f"Language: {lang_name}\n"
        f"Text: {text}\n"
        "Label:"
    )
    return system + "\n\n" + user

def build_prompt_taxi1500(lang: str, text: str) -> str:
    lang_name = LANG_CODE2NAME.get(lang, lang)
    labels = list(TAXI1500_LABEL2ID.keys())
    label_block = "\n".join(labels)
    system = (
        "You are a multilingual verse classifier.\n"
        "You must output exactly ONE label word from this closed set:\n"
        f"{label_block}\n\n"
        "Output rules:\n"
        "- Output exactly one label from the set above.\n"
        "- No extra words, punctuation, quotes, spaces, or newlines.\n\n"
        "Example:\n"
        "Language: English\n"
        "Text: Love your neighbor as yourself.\n"
        "Label: recommendation\n"
    )
    user = (
        f"Language: {lang_name}\n"
        f"Text: {text}\n"
        "Label:"
    )
    return system + "\n\n" + user


# ----------------------------
# Repair predictor (with args)
# ----------------------------
class RepairPredictor:
    def __init__(
        self,
        model_dir: str,
        device_map: str,
        dtype: str,
        use_chat_template: bool,
        max_new_tokens: int,
        temperature: float,
        max_tries: int,
    ):
        self.model_dir = model_dir
        self.device_map = device_map
        self.dtype = dtype
        self.use_chat_template = use_chat_template
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_tries = max_tries

        self.tokenizer = None
        self.model = None

    def ensure_loaded(self):
        if self.model is not None and self.tokenizer is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_dir,
            device_map=self.device_map,
            torch_dtype=_torch_dtype(self.dtype),
            trust_remote_code=True,
        )
        eos_id = _normalize_eos_id(self.tokenizer)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = eos_id

    def predict_label_text(self, prompt_text: str, allowed_labels: List[str]) -> Tuple[str, Dict[str, Any]]:
        self.ensure_loaded()
        eos_id = _normalize_eos_id(self.tokenizer)

        enc = encode_prompt(self.tokenizer, prompt_text, use_chat_template=self.use_chat_template)
        input_ids = enc["input_ids"]
        attn = enc.get("attention_mask", None)

        if hasattr(self.model, "device") and str(self.model.device) != "meta":
            input_ids = input_ids.to(self.model.device)
            if attn is not None:
                attn = attn.to(self.model.device)

        prompt_len = input_ids.shape[1]
        trie, max_lab_len = build_label_trie(self.tokenizer, allowed_labels)
        prefix_fn = make_prefix_allowed_tokens_fn(trie, prompt_len, eos_id)

        do_sample = self.temperature is not None and self.temperature > 0
        gen_max_new = max(self.max_new_tokens, max_lab_len + 2)

        last = ""
        for _ in range(self.max_tries):
            with torch.inference_mode():
                out = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    max_new_tokens=gen_max_new,
                    do_sample=do_sample,
                    temperature=self.temperature if do_sample else None,
                    top_p=1.0,
                    num_return_sequences=1,
                    return_dict_in_generate=True,
                    output_scores=True,
                    prefix_allowed_tokens_fn=prefix_fn,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=eos_id,
                )
            seq = out.sequences[0]
            gen_ids = seq[prompt_len:]
            last = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            lab = normalize_label_text(last)
            if lab in [x.lower() for x in allowed_labels]:
                cand_logits = score_label_candidates_token_logits(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt_input_ids=input_ids,
                    prompt_attention_mask=attn,
                    candidate_labels=allowed_labels,
                )
                return lab, cand_logits

        cand_logits = score_label_candidates_token_logits(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt_input_ids=input_ids,
            prompt_attention_mask=attn,
            candidate_labels=allowed_labels,
        )
        best = max(allowed_labels, key=lambda x: cand_logits[x]["sum_token_logits"])
        return best, cand_logits


def repredict_if_needed(
    obj: dict,
    dataset: str,
    lang: str,
    predictor: RepairPredictor,
    repair_enabled: bool,
) -> int:
    num_labels = NUM_LABELS[dataset]
    pred = get_pred_from_fields(obj, lang, num_labels=num_labels)
    if pred is not None:
        obj.setdefault("orig_pred", {})[lang] = int(pred)
        return int(pred)

    if not repair_enabled:
        raise RuntimeError(f"Invalid orig_pred and repair disabled, index={obj.get('index')} lang={lang}")

    if dataset == "xnli":
        premise = obj.get("premise", {}).get(lang)
        hypothesis = obj.get("hypothesis", {}).get(lang)
        if premise is None or hypothesis is None:
            raise RuntimeError(f"Missing premise/hypothesis for repair, index={obj.get('index')} lang={lang}")
        prompt = build_prompt_xnli(lang, premise, hypothesis)
        allowed = ["entailment", "neutral", "contradiction"]
        label_text, cand_logits = predictor.predict_label_text(prompt, allowed)
        pred_id = XNLI_LABEL2ID[label_text]
    elif dataset == "sib200":
        text = obj.get("text", {}).get(lang)
        if text is None:
            raise RuntimeError(f"Missing text for repair, index={obj.get('index')} lang={lang}")
        prompt = build_prompt_sib200(lang, text)
        allowed = list(SIB200_LABEL2ID.keys())
        label_text, cand_logits = predictor.predict_label_text(prompt, allowed)
        pred_id = SIB200_LABEL2ID[label_text]
    else:  # taxi1500
        text = obj.get("text", {}).get(lang)
        if text is None:
            raise RuntimeError(f"Missing text for repair, index={obj.get('index')} lang={lang}")
        prompt = build_prompt_taxi1500(lang, text)
        allowed = list(TAXI1500_LABEL2ID.keys())
        label_text, cand_logits = predictor.predict_label_text(prompt, allowed)
        pred_id = TAXI1500_LABEL2ID[label_text]

    obj.setdefault("orig_pred", {})[lang] = int(pred_id)
    obj.setdefault("orig_pred_text", {})[lang] = label_text
    obj.setdefault("orig_pred_candidate_logits", {})[lang] = cand_logits
    return int(pred_id)


def choose_target(num_labels: int, label: int, orig_pred: int, preferred: Optional[int] = None) -> int:
    excluded = {label, orig_pred}
    candidates = [c for c in range(num_labels) if c not in excluded]
    if not candidates:
        raise RuntimeError(f"No candidates left after excluding label={label}, orig_pred={orig_pred}")
    if preferred is not None and preferred in candidates:
        return preferred
    return random.choice(candidates)


def build_en_target_map(items: List[dict], dataset: str, predictor: RepairPredictor, repair_enabled: bool) -> Dict[int, int]:
    num_labels = NUM_LABELS[dataset]
    mp: Dict[int, int] = {}
    for obj in items:
        idx = obj.get("index", None)
        if idx is None:
            raise RuntimeError("Missing 'index' field in an item.")
        label = obj.get("label", None)
        if label is None:
            raise RuntimeError(f"Missing 'label' for index={idx}")

        orig_pred = repredict_if_needed(obj, dataset=dataset, lang="en", predictor=predictor, repair_enabled=repair_enabled)
        en_target = choose_target(num_labels=num_labels, label=label, orig_pred=orig_pred, preferred=None)
        mp[idx] = en_target
        obj["target_pred"] = {"en": en_target}
    return mp


def apply_targets_for_lang(
    items: List[dict],
    dataset: str,
    lang: str,
    en_target_map: Dict[int, int],
    predictor: RepairPredictor,
    repair_enabled: bool,
) -> None:
    num_labels = NUM_LABELS[dataset]
    for obj in items:
        idx = obj.get("index", None)
        if idx is None:
            raise RuntimeError("Missing 'index' field in an item.")
        label = obj.get("label", None)
        if label is None:
            raise RuntimeError(f"Missing 'label' for index={idx}")

        orig_pred = repredict_if_needed(obj, dataset=dataset, lang=lang, predictor=predictor, repair_enabled=repair_enabled)
        preferred = en_target_map.get(idx, None)
        tgt = choose_target(num_labels=num_labels, label=label, orig_pred=orig_pred, preferred=preferred)
        obj.setdefault("target_pred", {})[lang] = tgt


def process_split(
    data_root: Path,
    dataset: str,
    split: str,
    langs: List[str],
    predictor: RepairPredictor,
    repair_enabled: bool,
) -> None:
    split_dir = data_root / dataset / split
    if not split_dir.exists():
        print(f"[skip] Missing directory: {split_dir}")
        return

    en_path = split_dir / "en.jsonl"
    if not en_path.exists():
        print(f"[skip] Missing en.jsonl: {en_path}")
        return

    en_items = read_jsonl(en_path)
    en_target_map = build_en_target_map(en_items, dataset=dataset, predictor=predictor, repair_enabled=repair_enabled)
    write_jsonl_atomic(en_path, en_items)
    print(f"[ok] Wrote targets: {en_path}")

    for lang in langs:
        if lang == "en":
            continue
        path = split_dir / f"{lang}.jsonl"
        if not path.exists():
            print(f"[skip] Missing file: {path}")
            continue

        items = read_jsonl(path)
        apply_targets_for_lang(
            items,
            dataset=dataset,
            lang=lang,
            en_target_map=en_target_map,
            predictor=predictor,
            repair_enabled=repair_enabled,
        )
        write_jsonl_atomic(path, items)
        print(f"[ok] Wrote targets: {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, default="./data_qwen_pred_demo")
    ap.add_argument("--datasets", nargs="+", default=["xnli", "sib200", "taxi1500"])

    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    ap.add_argument("--langs", nargs="+", default=["en", "ar", "de", "ru", "sw", "vi", "zh"])

    ap.add_argument("--seed", type=int, default=569)

    ap.add_argument("--repair_enabled", action="store_true")
    ap.add_argument("--no_repair", dest="repair_enabled", action="store_false")
    ap.set_defaults(repair_enabled=True)

    ap.add_argument("--repair_model_dir", type=str, default="/root/autodl-tmp/model/gemma-2-9b-it")
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--use_chat_template", action="store_true")
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tries", type=int, default=50)

    args = ap.parse_args()

    random.seed(args.seed)

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise RuntimeError(f"data_root does not exist: {data_root.resolve()}")

    for ds in args.datasets:
        if ds not in NUM_LABELS:
            raise RuntimeError(f"Unknown dataset: {ds}")

    predictor = RepairPredictor(
        model_dir=args.repair_model_dir,
        device_map=args.device_map,
        dtype=args.dtype,
        use_chat_template=args.use_chat_template,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        max_tries=args.max_tries,
    )

    for dataset in args.datasets:
        for split in args.splits:
            process_split(
                data_root=data_root,
                dataset=dataset,
                split=split,
                langs=args.langs,
                predictor=predictor,
                repair_enabled=args.repair_enabled,
            )


if __name__ == "__main__":
    main()
