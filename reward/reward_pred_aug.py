import os
import re
import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList


LANGS_ALLOWED = ["en", "ar", "de", "ru", "sw", "vi", "zh"]

SIB200_LABEL2ID = {
    "science/technology": 0,
    "travel": 1,
    "politics": 2,
    "sports": 3,
    "health": 4,
    "entertainment": 5,
    "geography": 6,
}

XNLI_LABEL_SPEC = "0=entailment, 1=neutral, 2=contradiction"

_INT_RE = re.compile(r"(-?\d+)")


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def parse_first_int(s: str) -> Optional[int]:
    m = _INT_RE.search(s.strip())
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


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


class StopOnSubstrings(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings: List[str], start_len: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
        self.start_len = start_len

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        gen_ids = input_ids[0, self.start_len:]
        if gen_ids.numel() == 0:
            return False
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return any(st in text for st in self.stop_strings)


def encode_prompt(tokenizer, prompt_text: str, use_chat_template: bool = True) -> Dict[str, torch.Tensor]:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt_text}]
        enc = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
        return enc
    else:
        return tokenizer(prompt_text, return_tensors="pt", add_special_tokens=True)


def call_sample_and_logprob_local(
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int = 4,
    temperature: float = 0.0,
    stop: Optional[List[str]] = None,
    use_chat_template: bool = True,
) -> Dict[str, Any]:
    model.eval()

    enc = encode_prompt(tokenizer, prompt_text, use_chat_template=use_chat_template)
    input_ids = enc["input_ids"]

    if hasattr(model, "device") and str(model.device) != "meta":
        input_ids = input_ids.to(model.device)
        if "attention_mask" in enc:
            enc["attention_mask"] = enc["attention_mask"].to(model.device)

    prompt_len = input_ids.shape[1]

    stopping = None
    if stop:
        stopping = StoppingCriteriaList([StopOnSubstrings(tokenizer, stop_strings=stop, start_len=prompt_len)])

    do_sample = temperature is not None and temperature > 0

    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=enc.get("attention_mask", None),
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
            top_p=1.0,
            num_return_sequences=1,
            return_dict_in_generate=True,
            output_scores=True,
            stopping_criteria=stopping,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    seq = outputs.sequences[0]
    out_ids = seq[prompt_len:]
    out_tokens = out_ids.tolist()

    out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
    out_text_stripped = out_text.strip()

    token_logprobs: List[float] = []
    if outputs.scores is not None and len(outputs.scores) > 0:
        gen_len = min(len(outputs.scores), out_ids.numel())
        for t in range(gen_len):
            scores_t = outputs.scores[t][0]
            log_probs_t = F.log_softmax(scores_t, dim=-1)
            tok_id = int(out_ids[t].item())
            token_logprobs.append(float(log_probs_t[tok_id].item()))

    total_logprob = float(sum(token_logprobs)) if token_logprobs else None

    return {
        "raw_text": out_text,
        "text": out_text_stripped,
        "tokens": out_tokens,
        "token_logprobs": token_logprobs,
        "total_logprob": total_logprob,
    }


def _single_token_id(tokenizer, s: str) -> int:
    ids = tokenizer.encode(s, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"Label string {s!r} is not a single token for this tokenizer: {ids}")
    return ids[0]


def get_label_token_ids(tokenizer, task: str) -> Tuple[List[int], List[int]]:
    if task == "xnli":
        labels = [0, 1, 2]
    else:
        labels = [0, 1, 2, 3, 4, 5, 6]
    token_ids = [_single_token_id(tokenizer, str(x)) for x in labels]
    return labels, token_ids


def next_token_logits(model, tokenizer, prompt_text: str, use_chat_template: bool) -> torch.Tensor:
    enc = encode_prompt(tokenizer, prompt_text, use_chat_template=use_chat_template)
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc.get("attention_mask", None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(model.device)

    with torch.inference_mode():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits[0, -1, :]  # [vocab]
    return logits


def restricted_probs_from_logits(
    logits_vocab: torch.Tensor,
    label_ids: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    cand_logits = logits_vocab[label_ids]  # [C]
    cand_probs = F.softmax(cand_logits, dim=-1)
    return cand_logits, cand_probs


def predict_with_confidence(
    model,
    tokenizer,
    prompt_text: str,
    task: str,
    use_chat_template: bool,
    max_new_tokens: int = 1,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    labels, label_token_ids = get_label_token_ids(tokenizer, task)

    gen = call_sample_and_logprob_local(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        stop=None,
        use_chat_template=use_chat_template,
    )
    pred_from_text = parse_first_int(gen["text"])

    logits_vocab = next_token_logits(model, tokenizer, prompt_text, use_chat_template=use_chat_template)
    cand_logits, cand_probs = restricted_probs_from_logits(logits_vocab, label_token_ids)

    best_i = int(torch.argmax(cand_probs).item())
    pred_restricted = labels[best_i]
    conf_restricted = float(cand_probs[best_i].item())

    # If generation gave a weird output, fall back to restricted argmax
    pred_final = pred_from_text if pred_from_text in labels else pred_restricted

    # Build per-label logits (only small set, safe to store)
    logits_small = {str(labels[i]): float(cand_logits[i].item()) for i in range(len(labels))}
    probs_small = {str(labels[i]): float(cand_probs[i].item()) for i in range(len(labels))}

    conf_of_pred = float(probs_small[str(pred_final)])

    return {
        "pred": pred_final,
        "pred_text": gen["text"],
        "pred_total_logprob": gen["total_logprob"],
        "pred_token_logprobs": gen["token_logprobs"],
        "label_logits": logits_small,
        "label_probs": probs_small,
        "confidence": conf_of_pred,
    }





import math

def compute_rewards_flip_aug(
    pred: int,
    y_target: int,
    y_orig: int,
    label_logits: Dict[str, float],       # logits on x_hat
    orig_label_logits: Dict[str, float],  # logits on x (original)
    sigmoid_c: float = 5.5,
    sigmoid_k: float = 0.5,
) -> Dict[str, float]:
    # flip indicator
    r_flip = 1.0 if pred == y_target else 0.0

    # ----- flip gap: |z_{y'}(x_hat) - z_y(x_hat)|
    z_t_hat = float(label_logits.get(str(y_target), float("nan")))
    z_y_hat = float(label_logits.get(str(y_orig), float("nan")))
    gap_flip = float(abs(z_t_hat - z_y_hat)) if (z_t_hat == z_t_hat and z_y_hat == z_y_hat) else float("nan")

    # sigmoid(gap_flip - c)；reward_aug_flip
    if gap_flip == gap_flip:
        r_aug_flip = 1.0 / (1.0 + math.exp(-sigmoid_k * (gap_flip - sigmoid_c)))
    else:
        r_aug_flip = float("nan")

    # ----- noflip gap:  z_y(x) - z_y(x_hat)
    z_y_x = float(orig_label_logits.get(str(y_orig), float("nan")))
    gap_noflip = float(z_y_x - z_y_hat) if (z_y_hat == z_y_hat and z_y_x == z_y_x) else float("nan")

    #  sigmoid(c - gap_flip)；noflip reward
    if gap_noflip == gap_noflip:
        r_aug_noflip = (1.0 / (1.0 + math.exp(-sigmoid_k * (gap_noflip))) - 0.5) * 2
    else:
        r_aug_noflip = float("nan")

    return {
        "reward_flip": r_flip,

        "reward_aug_flip": r_aug_flip,
        "logit_gap_abs_flip": gap_flip,

        "reward_aug_noflip": r_aug_noflip,
        "logit_gap_abs_noflip": gap_noflip,
    }



def eval_one_example(
    task: str,
    lang: str,
    ex: Dict[str, Any],
    model,
    tokenizer,
    use_chat_template: bool,
) -> Dict[str, Any]:
    y_target = ex.get("target_pred", {}).get(lang, None)

    if task == "sib200":
        x_orig = ex["text"][lang]
        prompt_orig = build_prompt_sib200(lang, x_orig)
    else:
        premise = ex["premise"][lang]
        hypothesis = ex["hypothesis"][lang]
        x_orig = premise
        prompt_orig = build_prompt_xnli(lang, premise, hypothesis)

    orig_out = predict_with_confidence(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_orig,
        task=task,
        use_chat_template=use_chat_template,
        max_new_tokens=1,
        temperature=0.0,
    )
    y_orig = orig_out["pred"]

    # write original stats near orig_pred fields
    ex["orig_pred_runtime"] = ex.get("orig_pred_runtime", {})
    ex["orig_pred_runtime"][lang] = y_orig
    ex["orig_confidence"] = ex.get("orig_confidence", {})
    ex["orig_confidence"][lang] = orig_out["confidence"]
    # store only small logits/probs
    ex["orig_label_logits"] = ex.get("orig_label_logits", {})
    ex["orig_label_logits"][lang] = orig_out["label_logits"]
    ex["orig_label_probs"] = ex.get("orig_label_probs", {})
    ex["orig_label_probs"][lang] = orig_out["label_probs"]


    '''
    if y_target is not None:
        r = compute_rewards_flip_aug(
            pred=y_orig,
            y_target=int(y_target),
            y_orig=int(y_orig),
            label_logits=orig_out["label_logits"],
        )
        ex["orig_logit_gap_abs_to_target"] = ex.get("orig_logit_gap_abs_to_target", {})
        ex["orig_logit_gap_abs_to_target"][lang] = r["logit_gap_abs"]
        
    '''

    # counterfactuals
    cfs = ex.get("counterfactual", {}).get(lang, [])
    for cf in cfs:
        if task == "sib200":
            x_hat = cf["text"]
            prompt_cf = build_prompt_sib200(lang, x_hat)
        else:
            premise_hat = cf["text"]  # modified premise only
            prompt_cf = build_prompt_xnli(lang, premise_hat, hypothesis)

        cf_out = predict_with_confidence(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_cf,
            task=task,
            use_chat_template=use_chat_template,
            max_new_tokens=1,
            temperature=0.0,
        )

        cf["pred"] = cf_out["pred"]
        cf["pred_text"] = cf_out["pred_text"]
        cf["confidence"] = cf_out["confidence"]
        cf["label_logits"] = cf_out["label_logits"]
        cf["label_probs"] = cf_out["label_probs"]

        if y_target is not None:
            rr = compute_rewards_flip_aug(
                pred=int(cf_out["pred"]),
                y_target=int(y_target),
                y_orig=int(y_orig),
                label_logits=cf_out["label_logits"],      # x_hat logits
                orig_label_logits=orig_out["label_logits"]# x logits (needed for noflip)
            )

            cf["reward_flip"] = rr["reward_flip"]

            cf["reward_aug_flip"] = rr["reward_aug_flip"]
            cf["logit_gap_abs_flip"] = rr["logit_gap_abs_flip"]

            cf["reward_aug_noflip"] = rr["reward_aug_noflip"]
            cf["logit_gap_abs_noflip"] = rr["logit_gap_abs_noflip"]


    return ex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["xnli", "sib200"], required=True)
    ap.add_argument("--lang", choices=LANGS_ALLOWED, required=True)
    ap.add_argument("--input_jsonl", type=str, required=True)

    ap.add_argument("--local_model_dir", type=str, default="/root/autodl-tmp/model/Qwen3-8B")
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--local_files_only", action="store_true")
    ap.add_argument("--use_chat_template", action="store_true")

    ap.add_argument("--max_examples", type=int, default=1)
    args = ap.parse_args()

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

    in_path = Path(args.input_jsonl)
    rows: List[Dict[str, Any]] = []
    with in_path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if args.max_examples is not None and i >= args.max_examples:
                break
            rows.append(json.loads(line))

    for ex in rows:
        eval_one_example(args.task, args.lang, ex, model, tokenizer, use_chat_template=args.use_chat_template)

    # print the first processed example for inspection
    print(json.dumps(rows[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
