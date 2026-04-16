import os
import json
import argparse
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


LANGS_ALLOWED = ["en", "ar", "de", "ru", "sw", "vi", "zh", "bg", "el", "es", "hi", "th", "tr"]

LANG_CODE2NAME = {
    "en": "English",
    "ar": "Arabic",
    "de": "German",
    "ru": "Russian",
    "sw": "Swahili",
    "vi": "Vietnamese",
    "zh": "Chinese",
    "bg": "Bulgarian",
    "el": "Greek",
    "es": "Spanish",
    "hi": "Hindi",
    "th": "Thai",
    "tr": "Turkish",
}




SIB200_LABEL2ID = {
    "entertainment": 0,
    "geography": 1,
    "health": 2,     
    "politics": 3,
    "science/technology": 4,
    "sports": 5,
    "travel": 6,
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

DEFAULT_LIMITS = {
    ("xnli", "train"): 1500,
    ("xnli", "test"): 400,
    ("xnli", "validation"): 500,
    ("sib200", "train"): 1500,
    ("sib200", "test"): 204,
    ("sib200", "validation"): 99,
    ("taxi1500", "train"): 1500,
    ("taxi1500", "test"): 400,
    ("taxi1500", "validation"): 500,
}


# -----------------------------
# IO
# -----------------------------
def load_json_list(path: Path) -> List[Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def output_path(out_root: Path, task: str, split: str, lang: str) -> Path:
    return out_root / task / split / f"{lang}.jsonl"


def count_existing_lines(p: Path) -> int:
    if not p.exists():
        return 0
    with p.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


# -----------------------------
# Robust XNLI extraction
# -----------------------------
def _first_str_in_dict(d: Dict[str, Any]) -> Optional[str]:
    for v in d.values():
        if isinstance(v, str) and v.strip():
            return v
    return None


def _coerce_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return str(x)


def _get_lang_text(field: Any, srclang: str) -> str:
    if isinstance(field, dict):
        if srclang in field:
            return _coerce_text(field[srclang])
        if "en" in field:
            return _coerce_text(field["en"])
        s = _first_str_in_dict(field)
        if s is not None:
            return s
        try:
            return _coerce_text(next(iter(field.values())))
        except Exception:
            return ""

    if isinstance(field, list):
        if all(isinstance(t, str) for t in field):
            return " ".join([t for t in field if t is not None])
        return _coerce_text(field)

    return _coerce_text(field)


def extract_xnli_pair(ex: Any, srclang: str) -> Tuple[str, str]:
    if not isinstance(ex, dict):
        raise TypeError(f"XNLI example is not dict, got type={type(ex)} value={repr(ex)[:200]}")

    premise_field = ex.get("premise", "")
    premise = _get_lang_text(premise_field, srclang)

    hyp = ex.get("hypothesis", "")
    if isinstance(hyp, dict) and "language" in hyp and "translation" in hyp:
        langs = hyp.get("language", [])
        trans = hyp.get("translation", [])
        if isinstance(langs, list) and isinstance(trans, list):
            try:
                idx = langs.index(srclang)
                if 0 <= idx < len(trans):
                    hypothesis = _coerce_text(trans[idx])
                else:
                    hypothesis = _get_lang_text(hyp, srclang)
            except Exception:
                hypothesis = _get_lang_text(hyp, srclang)
        else:
            hypothesis = _get_lang_text(hyp, srclang)
    else:
        hypothesis = _get_lang_text(hyp, srclang)

    return premise, hypothesis


# -----------------------------
# Prompts
# -----------------------------




def build_prompt_xnli(lang_code: str, premise: str, hypothesis: str) -> str:
    lang_name = LANG_CODE2NAME.get(lang_code, lang_code)
    
    # System: Only role and core constraint
    system = (
        "You are a Multilingual Natural Language Inference classifier. /no_think"
        "Output exactly one word: entailment, neutral, or contradiction."
    )
    
    # User: Task definition, examples, and actual input
    user = (
        f"Classify the relationship between Premise and Hypothesis in {lang_name}.\n\n"
        
        "Labels:\n"
        "• entailment - Premise makes Hypothesis definitely true\n"
        "• contradiction - Premise makes Hypothesis definitely false\n"
        "• neutral - Not enough information to determine\n\n"
        
        "Example:\n"
        "Premise: A dog is running in the park.\n"
        "Hypothesis: An animal is running outdoors.\n"
        "Answer: entailment\n\n"
        
        "Now classify:\n"
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        "Answer: "
    )
    
    return system + "\n\n" + user


def build_prompt_sib200(lang_code: str, text: str) -> str:
    lang_name = LANG_CODE2NAME.get(lang_code, lang_code)
    labels = list(SIB200_LABEL2ID.keys())
    label_block = ", ".join(labels)
    
    # System: Only role and core constraint
    system = (
        "You are a multilingual topic classifier. /no_think"
        "Output exactly one topic label with no additional text."
    )
    
    # User: Task definition, labels, examples, and actual input
    user = (
        f"Classify the primary topic of this {lang_name} text.\n\n"
        
        f"Available topics: {label_block}\n\n"
        
        "Rules:\n"
        "• Choose the single best topic the text is mainly about\n"
        "• If multiple topics appear, pick the primary subject\n"
        "• Output only the label word\n\n"
        
        "Example:\n"
        "Text: The prime minister announced new election dates.\n"
        "Answer: politics\n\n"
        
        "Now classify:\n"
        f"Text: {text}\n"
        "Answer: "
    )
    
    return system + "\n\n" + user



def build_prompt_taxi1500(lang_code: str, text: str) -> str:
    lang_name = LANG_CODE2NAME.get(lang_code, lang_code)
    labels = list(TAXI1500_LABEL2ID.keys())
    label_block = ", ".join(labels)

    system = (
        "You are a multilingual verse classifier. /no_think\n"
        "Output exactly one label with no additional text."
    )

    user = (
        f"Classify the category of this {lang_name} verse/text.\n\n"
        f"Available labels: {label_block}\n\n"
        "Rules:\n"
        "• Choose the single best label\n"
        "• Output only the label word exactly as provided\n\n"
        "Example:\n"
        "Text: Love your neighbor as yourself.\n"
        "Answer: recommendation\n\n"
        "Now classify:\n"
        f"Text: {text}\n"
        "Answer: "
    )

    return system + "\n\n" + user









# -----------------------------
# Encode prompt (normalized)
# -----------------------------
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


def _normalize_eos_id(tokenizer) -> int:
    eos_id = tokenizer.eos_token_id
    # eos_token_id can be int or list[int] [web:1]
    if isinstance(eos_id, (list, tuple)):
        if len(eos_id) == 0:
            raise RuntimeError("tokenizer.eos_token_id is empty list/tuple")
        eos_id = eos_id[0]
    if not isinstance(eos_id, int):
        raise RuntimeError(f"tokenizer.eos_token_id is not int after normalize, got {type(eos_id)}")
    return eos_id


# -----------------------------
# Constrained decoding (Trie)
# -----------------------------
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
    out: List[List[int]] = []
    seen = set()
    for v in variants:
        ids = tokenizer.encode(v, add_special_tokens=False)
        if not ids:
            continue
        k = tuple(ids)
        if k not in seen:
            seen.add(k)
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
    """
    IMPORTANT FIX:
    Some Transformers versions call prefix_allowed_tokens_fn(batch_id, sent),
    where sent is a 1D tensor (not the full 2D batch). So we must support:
      - input_ids is 2D: [batch, cur_len]
      - input_ids is 1D: [cur_len]
      - input_ids is scalar: []
    [web:1]
    """
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
            # very defensive
            if isinstance(input_ids, int):
                seq = [input_ids]
            else:
                seq = list(input_ids)

        # prompt_len may be longer than current seq very early (extremely rare); clamp.
        cut = prompt_len if prompt_len <= len(seq) else len(seq)
        gen_prefix = seq[cut:]

        is_end, nxt = trie.next_tokens(gen_prefix)
        if nxt:
            return nxt
        if is_end:
            return [eos_token_id]
        return [eos_token_id]

    return _fn


def normalize_label_text(s: str) -> str:
    return (s or "").strip().lower()


# -----------------------------
# Candidate token-level logits
# -----------------------------
def score_label_candidates_token_logits(
    model,
    tokenizer,
    prompt_input_ids: torch.Tensor,         # [1, L]
    prompt_attention_mask: Optional[torch.Tensor],
    candidate_labels: List[str],
) -> Dict[str, Dict[str, Any]]:
    if not isinstance(prompt_input_ids, torch.Tensor) or prompt_input_ids.ndim != 2:
        raise TypeError(f"prompt_input_ids must be 2D tensor, got {type(prompt_input_ids)} shape={getattr(prompt_input_ids,'shape',None)}")

    device = prompt_input_ids.device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else _normalize_eos_id(tokenizer)

    prompt_ids = prompt_input_ids[0].tolist()
    prompt_len = len(prompt_ids)

    cand_token_ids: List[List[int]] = []
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

    result: Dict[str, Dict[str, Any]] = {}
    for i, lab in enumerate(candidate_labels):
        ids = cand_token_ids[i]
        tok_logits: List[float] = []
        for j, tok_id in enumerate(ids):
            pos = prompt_len + j
            tok_logits.append(float(logits[i, pos - 1, tok_id].item()))
        result[lab] = {
            "token_ids": ids,
            "token_logits": tok_logits,
            "sum_token_logits": float(sum(tok_logits)),
        }
    return result


# -----------------------------
# Constrained prediction
# -----------------------------
def predict_label_constrained(
    model,
    tokenizer,
    prompt_text: str,
    labels: List[str],
    max_new_tokens: int,
    temperature: float,
    use_chat_template: bool,
    max_tries: int = 50,
) -> Dict[str, Any]:
    model.eval()

    enc = encode_prompt(tokenizer, prompt_text, use_chat_template=use_chat_template)
    if "input_ids" not in enc or not isinstance(enc["input_ids"], torch.Tensor):
        raise TypeError(f"encode_prompt must return dict with tensor input_ids, got keys={list(enc.keys())}")

    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask", None)

    if hasattr(model, "device") and str(model.device) != "meta":
        input_ids = input_ids.to(model.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model.device)

    eos_id = _normalize_eos_id(tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = eos_id

    prompt_len = input_ids.shape[1]
    trie, max_label_tok_len = build_label_trie(tokenizer, labels)
    gen_max_new = max(max_new_tokens, max_label_tok_len + 2)

    prefix_fn = make_prefix_allowed_tokens_fn(trie=trie, prompt_len=prompt_len, eos_token_id=eos_id)
    do_sample = temperature is not None and temperature > 0

    last_text = ""
    for _ in range(max_tries):
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=gen_max_new,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=1.0,
                num_return_sequences=1,
                return_dict_in_generate=True,
                output_scores=True,
                prefix_allowed_tokens_fn=prefix_fn,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_id,
            )

        seq = outputs.sequences[0]
        out_ids = seq[prompt_len:]
        out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
        last_text = out_text

        lab = normalize_label_text(out_text)
        if lab in [l.lower() for l in labels]:
            return {
                "raw_text": out_text,
                "label_text": lab,
                "generated_token_ids": out_ids.tolist(),
            }

    cand_logits = score_label_candidates_token_logits(
        model=model,
        tokenizer=tokenizer,
        prompt_input_ids=input_ids,
        prompt_attention_mask=attention_mask,
        candidate_labels=labels,
    )
    best = max(labels, key=lambda x: cand_logits[x]["sum_token_logits"])
    return {
        "raw_text": last_text,
        "label_text": best,
        "generated_token_ids": [],
        "fallback_warning": "too_many_invalid_generations_fallback_to_best_candidate_logits",
    }


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["xnli", "sib200","taxi1500"], required=True)
    ap.add_argument("--split", choices=["train", "test", "validation"], required=True)
    ap.add_argument("--lang", choices=LANGS_ALLOWED, required=True)

    ap.add_argument("--data_dir", type=str, default="./data")
    ap.add_argument("--out_dir", type=str, default="./data_qwen_pred")

    ap.add_argument("--local_model_dir", type=str, default="Qwen/Qwen3-8B")
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--use_chat_template", action="store_true")

    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--max_tokens", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    if args.max_examples is None:
        args.max_examples = DEFAULT_LIMITS.get((args.task, args.split), 100)

    if args.dtype == "float16":
        torch_dtype = torch.float16
    elif args.dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    data_dir = Path(args.data_dir)
    out_root = Path(args.out_dir)
    out_file = output_path(out_root, args.task, args.split, args.lang)
    safe_mkdir(out_file.parent)

    if args.task == "xnli":
        fn = {
            "train": "xnli/training_set.json",
            "test": "xnli/test_set.json",
            "validation": "xnli/validation_set.json",
        }[args.split]
        data_path = data_dir / fn
    elif args.task == "sib200":
        fn = {
            "train": "sib200/sib200_train.json",
            "test": "sib200/sib200_test.json",
            "validation": "sib200/sib200_validation.json",
        }[args.split]
        data_path = data_dir / fn

    else:  # taxi1500
        fn = {
            "train": "taxi1500/taxi1500_train.json",
            "test": "taxi1500/taxi1500_test.json",
            "validation": "taxi1500/taxi1500_validation.json",
        }[args.split]
        data_path = data_dir / fn

    data = load_json_list(data_path)

    tokenizer = AutoTokenizer.from_pretrained(
        args.local_model_dir,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        token=os.environ.get("HF_TOKEN_PATH", None)
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.local_model_dir,
        device_map=args.device_map,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        token=os.environ.get("HF_TOKEN_PATH", None)
    )

    eos_id = _normalize_eos_id(tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = eos_id

    already = count_existing_lines(out_file) if args.resume else 0

    written = 0
    with out_file.open("a", encoding="utf-8") as wf:
        for i, ex in enumerate(data):
            if args.resume and i < already:
                continue
            if written >= args.max_examples:
                break

            try:
                if args.task == "xnli":
                    premise, hypothesis = extract_xnli_pair(ex, args.lang)
                    prompt_text = build_prompt_xnli(args.lang, premise, hypothesis)

                    labels = ["entailment", "neutral", "contradiction"]
                    pred_pack = predict_label_constrained(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_text=prompt_text,
                        labels=labels,
                        max_new_tokens=args.max_tokens,
                        temperature=args.temperature,
                        use_chat_template=args.use_chat_template,
                    )
                    pred_label = pred_pack["label_text"]
                    pred_id = XNLI_LABEL2ID.get(pred_label, None)

                    enc = encode_prompt(tokenizer, prompt_text, use_chat_template=args.use_chat_template)
                    if hasattr(model, "device") and str(model.device) != "meta":
                        enc["input_ids"] = enc["input_ids"].to(model.device)
                        if "attention_mask" in enc:
                            enc["attention_mask"] = enc["attention_mask"].to(model.device)

                    cand_logits = score_label_candidates_token_logits(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_input_ids=enc["input_ids"],
                        prompt_attention_mask=enc.get("attention_mask", None),
                        candidate_labels=labels,
                    )

                    out_obj = {
                        "index": ex.get("index", i) if isinstance(ex, dict) else i,
                        "premise": {args.lang: premise},
                        "hypothesis": {args.lang: hypothesis},
                        "label": ex.get("label", None) if isinstance(ex, dict) else None,
                        "orig_pred": {args.lang: pred_id},
                        "orig_pred_text": {args.lang: pred_label},
                        "orig_pred_candidate_logits": {args.lang: cand_logits},
                    }
                    if "fallback_warning" in pred_pack:
                        out_obj.setdefault("warning", {})[args.lang] = pred_pack["fallback_warning"]

                elif args.task == "sib200":
                    if not isinstance(ex, dict):
                        raise TypeError(f"SIB200 example is not dict, got type={type(ex)} value={repr(ex)[:200]}")

                    text = ex["text"][args.lang] if isinstance(ex.get("text"), dict) and args.lang in ex["text"] else str(ex.get("text", ""))
                    prompt_text = build_prompt_sib200(args.lang, text)

                    labels = list(SIB200_LABEL2ID.keys())
                    pred_pack = predict_label_constrained(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_text=prompt_text,
                        labels=labels,
                        max_new_tokens=args.max_tokens,
                        temperature=args.temperature,
                        use_chat_template=args.use_chat_template,
                    )
                    pred_label = pred_pack["label_text"]
                    pred_id = SIB200_LABEL2ID.get(pred_label, None)

                    enc = encode_prompt(tokenizer, prompt_text, use_chat_template=args.use_chat_template)
                    if hasattr(model, "device") and str(model.device) != "meta":
                        enc["input_ids"] = enc["input_ids"].to(model.device)
                        if "attention_mask" in enc:
                            enc["attention_mask"] = enc["attention_mask"].to(model.device)

                    cand_logits = score_label_candidates_token_logits(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_input_ids=enc["input_ids"],
                        prompt_attention_mask=enc.get("attention_mask", None),
                        candidate_labels=labels,
                    )

                    out_obj = {
                        "index": ex.get("index", i),
                        "text": {args.lang: text},
                        "label": ex.get("label", None),
                        "orig_pred": {args.lang: pred_id},
                        "orig_pred_text": {args.lang: pred_label},
                        "orig_pred_candidate_logits": {args.lang: cand_logits},
                    }
                    if "fallback_warning" in pred_pack:
                        out_obj.setdefault("warning", {})[args.lang] = pred_pack["fallback_warning"]
                        
                else:  # taxi1500
                    if not isinstance(ex, dict):
                        raise TypeError(f"TAXI1500 example is not dict, got type={type(ex)} value={repr(ex)[:200]}")

                    # taxi1500 格式：ex["text"] 是 dict，包含 en/ar/de/ru/sw/vi/zh
                    text = ex["text"][args.lang] if isinstance(ex.get("text"), dict) and args.lang in ex["text"] else str(ex.get("text", ""))
                    prompt_text = build_prompt_taxi1500(args.lang, text)

                    labels = list(TAXI1500_LABEL2ID.keys())
                    pred_pack = predict_label_constrained(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_text=prompt_text,
                        labels=labels,
                        max_new_tokens=args.max_tokens,
                        temperature=args.temperature,
                        use_chat_template=args.use_chat_template,
                    )

                    pred_label = pred_pack["label_text"].strip()
                    pred_id = TAXI1500_LABEL2ID.get(pred_label, None)

                    enc = encode_prompt(tokenizer, prompt_text, use_chat_template=args.use_chat_template)
                    if hasattr(model, "device") and str(model.device) != "meta":
                        enc["input_ids"] = enc["input_ids"].to(model.device)
                        if "attention_mask" in enc:
                            enc["attention_mask"] = enc["attention_mask"].to(model.device)

                    cand_logits = score_label_candidates_token_logits(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_input_ids=enc["input_ids"],
                        prompt_attention_mask=enc.get("attention_mask", None),
                        candidate_labels=labels,
                    )

                    out_obj = {
                        "index": ex.get("index", i),
                        "id": ex.get("id", None),
                        "text": {args.lang: text},
                        "label": ex.get("label", None),
                        "orig_pred": {args.lang: pred_id},
                        "orig_pred_text": {args.lang: pred_label},
                        "orig_pred_candidate_logits": {args.lang: cand_logits},
                    }
                    if "fallback_warning" in pred_pack:
                        out_obj.setdefault("warning", {})[args.lang] = pred_pack["fallback_warning"]

                wf.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                wf.flush()
                written += 1

            except Exception as e:
                err_obj = {
                    "index": ex.get("index", i) if isinstance(ex, dict) else i,
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "task": args.task,
                    "split": args.split,
                    "lang": args.lang,
                }
                wf.write(json.dumps(err_obj, ensure_ascii=False) + "\n")
                wf.flush()
                written += 1

    print(f"[done] wrote {written} lines to {out_file}")


if __name__ == "__main__":
    main()
