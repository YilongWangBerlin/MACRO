import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import math
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM


LANGS_ALLOWED = ["en", "ar", "de", "ru", "sw", "vi", "zh", "bg", "el", "es", "hi", "th", "tr"]

SIB200_LABELS = [
    "entertainment",
    "geography",
    "health",
    "politics",
    "science/technology",
    "sports",
    "travel",
]

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

XNLI_LABELS = ["entailment", "neutral", "contradiction"]

TAXI1500_LABELS = ["recommendation", "faith", "violence", "grace", "sin", "description"]
TAXI1500_LABEL2ID = {k: i for i, k in enumerate(TAXI1500_LABELS)}

# -----------------------------
# Prompts
# -----------------------------
def build_prompt_xnli(lang_code: str, premise: str, hypothesis: str) -> str:
    lang_name = LANG_CODE2NAME.get(lang_code, lang_code)
    system = (
        "You are a Multilingual Natural Language Inference classifier. /no_think"
        "Output exactly one word: entailment, neutral, or contradiction."
    )
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
    label_block = ", ".join(list(SIB200_LABEL2ID.keys()))
    system = (
        "You are a multilingual topic classifier. /no_think"
        "Output exactly one topic label with no additional text."
    )
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
    label_block = ", ".join(TAXI1500_LABELS)
    system = (
        "You are a multilingual verse classifier. /no_think"
        "Output exactly one label with no additional text."
    )
    user = (
        f"Classify the category of this {lang_name} verse/text.\n\n"
        f"Available labels: {label_block}\n\n"
        "Rules:\n"
        "• Choose the single best label\n"
        "• Output only the label word\n\n"
        "Example:\n"
        "Text: Love your neighbor as yourself.\n"
        "Answer: recommendation\n\n"
        "Now classify:\n"
        f"Text: {text}\n"
        "Answer: "
    )
    return system + "\n\n" + user



# -----------------------------
# Helpers
# -----------------------------
def get_labels(task: str) -> List[str]:
    if task == "xnli":
        return XNLI_LABELS
    if task == "sib200":
        return SIB200_LABELS
    if task == "taxi1500":
        return TAXI1500_LABELS
    raise ValueError(f"Unknown task: {task}")



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

    if isinstance(enc, dict):
        return enc
    if hasattr(enc, "input_ids"):
        out = {"input_ids": enc.input_ids}
        if hasattr(enc, "attention_mask"):
            out["attention_mask"] = enc.attention_mask
        return out
    raise TypeError(f"Unexpected encode output: {type(enc)}")


def _normalize_eos_id(tokenizer) -> int:
    eos_id = tokenizer.eos_token_id
    if isinstance(eos_id, (list, tuple)):
        if len(eos_id) == 0:
            raise RuntimeError("tokenizer.eos_token_id is empty list/tuple")
        eos_id = eos_id[0]
    if not isinstance(eos_id, int):
        raise RuntimeError(f"tokenizer.eos_token_id is not int, got {type(eos_id)}")
    return eos_id


def normalize_label_text(s: str) -> str:
    return (s or "").strip().lower()


def _first_token_id_of_text(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"Text {text!r} tokenized to empty.")
    return int(ids[0])


# -----------------------------
# Step-0 logits for label variants
# -----------------------------
def get_step0_label_first_token_logits(
    model,
    tokenizer,
    enc: Dict[str, torch.Tensor],
    labels: List[str],
    use_raw_logits_if_available: bool = True,
) -> Tuple[Dict[str, float], str]:
    """
    Fixed: use the FIRST token of the PLAIN label string (no leading space/newline).
    """
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask", None)

    eos_id = _normalize_eos_id(tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = eos_id

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=1,
        do_sample=False,
        temperature=None,
        top_p=1.0,
        num_return_sequences=1,
        return_dict_in_generate=True,
        output_scores=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_id,
    )
    if use_raw_logits_if_available:
        gen_kwargs["output_logits"] = True
        
    #print("=== PRED GENERATION KWARGS ===")
    #print(gen_kwargs)
    #rint("============================")

    with torch.inference_mode():
        try:
            out = model.generate(**gen_kwargs)
        except TypeError:
            gen_kwargs.pop("output_logits", None)
            out = model.generate(**gen_kwargs)

    if use_raw_logits_if_available and hasattr(out, "logits") and out.logits is not None:
        step0 = out.logits[0][0]  # [vocab]
        source = "logits"
    else:
        step0 = out.scores[0][0]
        source = "scores"

    label_logits: Dict[str, float] = {}
    for lb in labels:
        tid = _first_token_id_of_text(tokenizer, lb)   # <-- plain label only
        label_logits[lb] = float(step0[tid].item())

    return label_logits, source



# -----------------------------
# Trie constrained decoding
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
    #variants = [label]

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


# -----------------------------
# Prediction (constrained pred, step0 logits for reward)
# -----------------------------
def predict_with_confidence_constrained(
    model,
    tokenizer,
    prompt_text: str,
    task: str,
    use_chat_template: bool,
    max_new_tokens: int = 8,
    temperature: float = 1.0,
    max_tries: int = 50,
    debug_print: bool = False,
) -> Dict[str, Any]:
    labels = get_labels(task)
    model.eval()

    enc = encode_prompt(tokenizer, prompt_text, use_chat_template=use_chat_template)
    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask", None)

    if hasattr(model, "device") and str(model.device) != "meta":
        enc["input_ids"] = input_ids.to(model.device)
        if attention_mask is not None:
            enc["attention_mask"] = attention_mask.to(model.device)

    eos_id = _normalize_eos_id(tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = eos_id

    prompt_len = enc["input_ids"].shape[1]
    trie, max_label_tok_len = build_label_trie(tokenizer, labels)
    gen_max_new = max(max_new_tokens, max_label_tok_len + 2)
    prefix_fn = make_prefix_allowed_tokens_fn(trie=trie, prompt_len=prompt_len, eos_token_id=eos_id)
    do_sample = temperature is not None and temperature > 0

    last_text = ""
    gen_token_ids: List[int] = []
    pred_label: Optional[str] = None

    for _ in range(max_tries):
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc.get("attention_mask", None),
                max_new_tokens=gen_max_new,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=1.0,
                num_return_sequences=1,
                return_dict_in_generate=True,
                output_scores=False,
                prefix_allowed_tokens_fn=prefix_fn,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_id,
            )

        seq = outputs.sequences[0]
        out_ids = seq[prompt_len:]
        out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
        last_text = out_text
        gen_token_ids = out_ids.tolist()

        
        #print("[constrained out_text]", repr(out_text))
        #print("[constrained out_ids]", gen_token_ids)
        pieces = [tokenizer.decode([tid], skip_special_tokens=False) for tid in gen_token_ids]
        #print("[constrained token pieces]")
        #for tid, p in zip(gen_token_ids, pieces):
        #    print(f"{tid}\t{p!r}")

        lab = normalize_label_text(out_text)
        if lab in [l.lower() for l in labels]:
            pred_label = lab
            break

    # Step-0 logits (variant-aware) for label_probs/confidence + reward gaps

    label_logits, score_source = get_step0_label_first_token_logits(
        model=model,
        tokenizer=tokenizer,
        enc=enc,
        labels=labels,
        use_raw_logits_if_available=True,
    )


    cand = torch.tensor([label_logits[lb] for lb in labels], dtype=torch.float32)
    probs = F.softmax(cand, dim=-1).detach().cpu().tolist()
    label_probs = {labels[i]: float(probs[i]) for i in range(len(labels))}

    warning = None
    if pred_label is None:
        pred_label = max(labels, key=lambda x: label_logits[x])
        warning = "too_many_invalid_generations_fallback_to_step0_variant_first_token_argmax"

    return {
        "pred": pred_label,
        "pred_text": last_text.strip(),
        "score_source": f"constrained_generate + step0_{score_source}_variant_max",
        "label_logits": label_logits,
        "label_probs": label_probs,
        "confidence": float(label_probs.get(pred_label, 0.0)),
        "generated_token_ids": gen_token_ids,
        **({"warning": warning} if warning else {}),
    }


# -----------------------------
# Reward
# -----------------------------
def compute_rewards_flip_aug(
    pred: str,
    y_orig: str,
    y_target: str,                 # NEW
    label_logits: Dict[str, float],       # step-0 (variant-max) first-token logits on x_hat
    orig_label_logits: Dict[str, float],  # step-0 first-token logits on x (from file)
    sigmoid_k: float = 1.0,
) -> Dict[str, float]:
    pred = str(pred).strip().lower()
    y_orig = str(y_orig).strip().lower()

    #r_flip = 1.0 if pred != y_orig else 0.0
    r_flip = 1.0 if pred == y_target else 0.0

    # NEW: “toward target” logit gap on x_hat
    z_tgt_hat = float(label_logits.get(y_target, float("nan")))
    z_orig_hat = float(label_logits.get(y_orig, float("nan")))
    gap_flip = float(z_tgt_hat - z_orig_hat) if (z_tgt_hat == z_tgt_hat and z_orig_hat == z_orig_hat) else float("nan")
    r_aug_flip = 1.0 / (1.0 + math.exp(-sigmoid_k * gap_flip)) if (gap_flip == gap_flip) else float("nan")

    # keep your old “no-flip/stability” term (still about preserving orig)
    z_orig_x = float(orig_label_logits.get(y_orig, float("nan")))
    gap_noflip = float(z_orig_x - z_orig_hat) if (z_orig_x == z_orig_x and z_orig_hat == z_orig_hat) else float("nan")
    r_aug_noflip = 1.0 / (1.0 + math.exp(-sigmoid_k * gap_noflip)) if (gap_noflip == gap_noflip) else float("nan")

    return {
        "reward_flip": r_flip,
        "reward_aug_flip": r_aug_flip,
        "logit_gap_abs_flip": gap_flip,
        "reward_aug_noflip": r_aug_noflip,
        "logit_gap_abs_noflip": gap_noflip,
    }


# -----------------------------
# Read orig logits (compat for old/new file formats)
# -----------------------------
def read_orig_label_logits_first_token(ex: Dict[str, Any], lang: str) -> Dict[str, float]:
    """
    Returns dict[label_lower -> first_token_logit] for orig x from saved file.
    Supports:
      - old: info["token_logits"][0]
      - new variants: info["variants"]["space"]["token_logits"][0] (preferred),
                      else best_variant, else old.
    """
    out: Dict[str, float] = {}
    orig_cands = ex.get("orig_pred_candidate_logits", {}).get(lang, {})
    if not isinstance(orig_cands, dict):
        return out

    for lb, info in orig_cands.items():
        lb_norm = str(lb).strip().lower()
        if not isinstance(info, dict):
            continue

        # new variants format
        if "variants" in info and isinstance(info["variants"], dict):
            v_space = info["variants"].get("space", None)
            if isinstance(v_space, dict):
                tl = v_space.get("token_logits", None)
                if isinstance(tl, list) and len(tl) > 0:
                    out[lb_norm] = float(tl[0])
                    continue

            bn = info.get("best_variant", None)
            if isinstance(bn, str) and bn in info["variants"]:
                tl = info["variants"][bn].get("token_logits", None)
                if isinstance(tl, list) and len(tl) > 0:
                    out[lb_norm] = float(tl[0])
                    continue

        # old format
        tl = info.get("token_logits", None)
        if isinstance(tl, list) and len(tl) > 0:
            out[lb_norm] = float(tl[0])

    return out


# -----------------------------
# Eval one example
# -----------------------------
def eval_one_example(
    task: str,
    lang: str,
    ex: Dict[str, Any],
    model,
    tokenizer,
    use_chat_template: bool,
    debug_print: bool = False,
) -> Dict[str, Any]:
    y_orig = ex.get("orig_pred_text", {}).get(lang, None)
    if y_orig is None:
        raise KeyError(f"Missing ex['orig_pred_text'][{lang}]")
    y_orig = str(y_orig).strip().lower()
    
    # NEW: target label text
    labels = get_labels(task)
    target_id = ex.get("target_pred", {}).get(lang, None)
    if target_id is None:
        y_target = str(ex.get("target_pred_text", {}).get(lang, "")).strip().lower()
        if not y_target:
            raise KeyError(f"Missing ex['target_pred'][{lang}] (and no target_pred_text fallback)")
    else:
        y_target = str(labels[int(target_id)]).strip().lower()

    orig_label_logits_first = read_orig_label_logits_first_token(ex, lang)

    cfs = ex.get("counterfactual", {}).get(lang, [])
    if not isinstance(cfs, list):
        return ex

    hypothesis = None
    if task == "xnli":
        hypothesis = ex["hypothesis"][lang]

    for cf in cfs:
        if task == "sib200":
            prompt_cf = build_prompt_sib200(lang, cf["text"])
        elif task == "taxi1500":
            prompt_cf = build_prompt_taxi1500(lang, cf["text"])
        else:  # xnli
            prompt_cf = build_prompt_xnli(lang, cf["text"], hypothesis)


        cf_out = predict_with_confidence_constrained(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_cf,
            task=task,
            use_chat_template=use_chat_template,
            max_new_tokens=8,
            temperature=1.0,
            max_tries=50,
            debug_print=debug_print,
        )

        cf["pred"] = cf_out["pred"]
        cf["pred_text"] = cf_out.get("pred_text", "")
        cf["confidence"] = cf_out.get("confidence", float("nan"))
        cf["label_logits"] = cf_out.get("label_logits", {})
        cf["label_probs"] = cf_out.get("label_probs", {})
        cf["score_source"] = cf_out.get("score_source", "")
        if "warning" in cf_out:
            cf["warning"] = cf_out["warning"]

        # optional: keep debug info (can be large)
        # cf["label_variant_debug"] = cf_out.get("label_variant_debug", {})

        rr = compute_rewards_flip_aug(
            pred=str(cf_out["pred"]).strip().lower(),
            y_orig=str(y_orig).strip().lower(),
            y_target=str(y_target).strip().lower(),  # NEW
            label_logits={k.lower(): float(v) for k, v in (cf_out.get("label_logits") or {}).items()},
            orig_label_logits={k.lower(): float(v) for k, v in orig_label_logits_first.items()},
        )
        cf["reward_flip"] = rr["reward_flip"]
        cf["reward_aug_flip"] = rr["reward_aug_flip"]
        cf["logit_gap_abs_flip"] = rr["logit_gap_abs_flip"]
        cf["reward_aug_noflip"] = rr["reward_aug_noflip"]
        cf["logit_gap_abs_noflip"] = rr["logit_gap_abs_noflip"]

    return ex


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["xnli", "sib200", "taxi1500"], required=True)
    ap.add_argument("--lang", choices=LANGS_ALLOWED, required=True)
    ap.add_argument("--input_jsonl", type=str, required=True)

    ap.add_argument("--local_model_dir", type=str, default="/root/autodl-tmp/model/Qwen3-8B")
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--use_chat_template", action="store_true")

    ap.add_argument("--max_examples", type=int, default=1)
    ap.add_argument("--debug_print", action="store_true")

    args = ap.parse_args()
    
    print("arg dict:", vars(args), flush=True)


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

    eos_id = _normalize_eos_id(tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = eos_id

    in_path = Path(args.input_jsonl)
    rows: List[Dict[str, Any]] = []
    with in_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.max_examples is not None and i >= args.max_examples:
                break
            rows.append(json.loads(line))

    for ex in rows:
        eval_one_example(
            task=args.task,
            lang=args.lang,
            ex=ex,
            model=model,
            tokenizer=tokenizer,
            use_chat_template=args.use_chat_template,
            debug_print=args.debug_print,
        )

    print(json.dumps(rows[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
