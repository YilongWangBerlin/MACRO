import json
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import math
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList


LANGS_ALLOWED = ["en", "ar", "de", "ru", "sw", "vi", "zh"]

SIB200_LABELS = [
    "entertainment",
    "geography",
    "health",
    "politics",
    "science/technology",
    "sports",
    "travel",
]

XNLI_LABELS = ["entailment", "neutral", "contradiction"]


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def build_prompt_xnli(lang: str, premise: str, hypothesis: str) -> str:
    system = (
        "You are a multilingual natural language inference (NLI) classifier./no_think\n"
        "Task: Given a Premise and a Hypothesis in the SAME language, output their relation label.\n"
        "Labels (output exactly ONE word):\n"
        "entailment\n"
        "neutral\n"
        "contradiction\n"
        "Rules:\n"
        "- Output MUST be exactly one label word from the set above.\n"
        "- Do NOT output punctuation, extra words, spaces, or newlines."
    )

    user = (
        f"Language: {lang}\n"
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        "Label: "
    )
    return system + "\n\n" + user


def build_prompt_sib200(lang: str, text: str) -> str:
    system = (
        "You are a multilingual topic classifier for short news sentences./no_think\n"
        "Assign exactly one topic LABEL WORD to the given news text.\n"
        "Allowed labels (output exactly ONE word/phrase):\n"
        + "\n".join(SIB200_LABELS) + "\n"
        "Guidelines:\n"
        "- Choose the single best topic that the text is mainly about.\n"
        "- If multiple topics appear, pick the primary event/subject.\n"
        "- If unsure, pick the closest single topic.\n\n"
        "Output rules:\n"
        "- Output EXACTLY ONE label from the allowed set.\n"
        "- Do NOT output punctuation, extra words, spaces, or newlines."
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


def get_labels(task: str) -> List[str]:
    return XNLI_LABELS if task == "xnli" else SIB200_LABELS


def _label_first_token_id(tokenizer, label: str) -> int:
    """
    Labels may be multi-token (e.g. 'science/technology').
    We only use the FIRST token id as the label id.
    """
    ids = tokenizer.encode(label, add_special_tokens=False)
    if len(ids) < 1:
        raise ValueError(f"Label string {label!r} tokenized to empty.")
    return ids[0]


def normalize_pred_text(s: str) -> str:
    return (s or "").strip().lower()


def map_pred_text_to_label(pred_text: str, labels: List[str]) -> Optional[str]:
    """
    Loose mapping for debugging only.
    The final pred used for rewards is taken from restricted logits argmax.
    """
    t = normalize_pred_text(pred_text)
    if t in labels:
        return t
    for lb in labels:
        if lb in t:
            return lb
    return None


def call_sample_and_logprob_local(
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int = 8,
    temperature: float = 0.0,
    stop: Optional[List[str]] = None,
    use_chat_template: bool = True,
    # return extra info from generate()
    return_token_logits: bool = True,
    topk_per_step: int = 0,
    restricted_token_ids: Optional[List[int]] = None,
    prefer_raw_logits: bool = True,
) -> Dict[str, Any]:
    """
    Run model.generate() once and return:
    - generated text
    - per-step logprobs for generated tokens
    - per-step token logits for generated tokens (optional)
    - per-step restricted logits for specified token ids (optional)
    Notes:
    - If the transformers version supports output_logits, we try to return raw logits;
      otherwise we fall back to outputs.scores.
    """
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

    gen_kwargs = dict(
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
    if prefer_raw_logits:
        gen_kwargs["output_logits"] = True

    with torch.inference_mode():
        try:
            outputs = model.generate(**gen_kwargs)
        except TypeError:
            gen_kwargs.pop("output_logits", None)
            outputs = model.generate(**gen_kwargs)

    seq = outputs.sequences[0]
    out_ids = seq[prompt_len:]
    out_tokens = out_ids.tolist()

    out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
    out_text_stripped = out_text.strip()

    # Pick score source for each generation step
    # - outputs.logits if available (raw)
    # - else outputs.scores (may be processed)
    step_scores = None
    score_source = None
    if prefer_raw_logits and hasattr(outputs, "logits") and outputs.logits is not None:
        step_scores = outputs.logits
        score_source = "logits"
    else:
        step_scores = outputs.scores
        score_source = "scores"

    token_logprobs: List[float] = []
    token_logits: List[float] = []
    step_topk: List[List[Tuple[int, float]]] = []
    restricted_step_logits: List[Dict[int, float]] = []

    if step_scores is not None and len(step_scores) > 0:
        gen_len = min(len(step_scores), out_ids.numel())
        for t in range(gen_len):
            scores_t = step_scores[t][0]  # (vocab,)
            tok_id = int(out_ids[t].item())

            log_probs_t = F.log_softmax(scores_t, dim=-1)
            token_logprobs.append(float(log_probs_t[tok_id].item()))

            if return_token_logits:
                token_logits.append(float(scores_t[tok_id].item()))

            if topk_per_step and topk_per_step > 0:
                topv, topi = torch.topk(scores_t, k=min(int(topk_per_step), scores_t.numel()))
                step_topk.append([(int(i.item()), float(v.item())) for v, i in zip(topv, topi)])

            if restricted_token_ids is not None:
                d = {}
                for rid in restricted_token_ids:
                    d[int(rid)] = float(scores_t[int(rid)].item())
                restricted_step_logits.append(d)

    total_logprob = float(sum(token_logprobs)) if token_logprobs else None
    total_logit = float(sum(token_logits)) if token_logits else None

    return {
        "raw_text": out_text,
        "text": out_text_stripped,
        "tokens": out_tokens,
        "token_logprobs": token_logprobs,
        "total_logprob": total_logprob,
        "token_logits": token_logits,
        "total_logit": total_logit,
        "score_source": score_source,
        "step_topk": step_topk,
        "restricted_step_logits": restricted_step_logits,
    }


def predict_with_confidence(
    model,
    tokenizer,
    prompt_text: str,
    task: str,
    use_chat_template: bool,
    max_new_tokens: int = 8,
    temperature: float = 0.0,
) -> Dict[str, Any]:
    """
    No teacher-forcing / no extra forward pass.
    We compute label logits from generate() step-0 scores for the FIRST token of each label.
    """
    labels = get_labels(task)
    label_first_token_ids = [_label_first_token_id(tokenizer, lb) for lb in labels]

    gen = call_sample_and_logprob_local(
        model=model,
        tokenizer=tokenizer,
        prompt_text=prompt_text,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        stop=None,
        use_chat_template=use_chat_template,
        restricted_token_ids=label_first_token_ids,  # capture label token logits at each step
        prefer_raw_logits=True,
    )

    pred_from_text = map_pred_text_to_label(gen["text"], labels)

    # Step-0 restricted logits (next-token distribution right after the prompt)
    logits_small: Dict[str, float] = {lb: float("nan") for lb in labels}
    if gen["restricted_step_logits"] and len(gen["restricted_step_logits"]) > 0:
        step0 = gen["restricted_step_logits"][0]  # dict: token_id -> logit/score
        for i, lb in enumerate(labels):
            tid = int(label_first_token_ids[i])
            if tid in step0:
                logits_small[lb] = float(step0[tid])

    # Convert to probs via softmax over labels (heuristic but consistent with logits_small)
    cand = torch.tensor([logits_small[lb] for lb in labels], dtype=torch.float32)
    cand_probs = F.softmax(cand, dim=-1).detach().cpu().tolist()
    probs_small = {labels[i]: float(cand_probs[i]) for i in range(len(labels))}

    best_i = int(torch.argmax(torch.tensor(cand_probs)).item())
    pred_restricted = labels[best_i]

    # IMPORTANT: use restricted argmax as final pred to avoid mismatch
    pred_final = pred_restricted
    conf_of_pred = float(probs_small[pred_final])

    return {
        "pred": pred_final,
        "pred_text": gen["text"],  # keep generated text for debugging
        "pred_from_text": pred_from_text,
        "pred_total_logprob": gen["total_logprob"],
        "pred_token_logprobs": gen["token_logprobs"],
        "pred_token_logits": gen["token_logits"],
        "score_source": gen["score_source"],
        "label_logits": logits_small,  # FIRST-token scores at step-0
        "label_probs": probs_small,
        "confidence": conf_of_pred,
        "restricted_best": pred_restricted,
        "restricted_conf": float(probs_small[pred_restricted]),
    }


def compute_rewards_flip_aug(
    pred: str,
    y_orig: str,
    label_logits: Dict[str, float],       # scores on x_hat (FIRST token only)
    orig_label_logits: Dict[str, float],  # scores on x (FIRST token only, from file)
    sigmoid_k: float = 1.0,
) -> Dict[str, float]:
    """
    Reward design using FIRST-token logits only (no teacher forcing):
    - Flip success: pred != y_orig
    - Flip gap: z_pred(x_hat) - z_orig(x_hat)
    - Noflip gap: z_orig(x) - z_orig(x_hat)
    """
    pred = str(pred).strip().lower()
    y_orig = str(y_orig).strip().lower()

    r_flip = 1.0 if pred != y_orig else 0.0

    z_pred_hat = float(label_logits.get(pred, float("nan")))
    z_orig_hat = float(label_logits.get(y_orig, float("nan")))
    gap_flip = float(z_pred_hat - z_orig_hat) if (z_pred_hat == z_pred_hat and z_orig_hat == z_orig_hat) else float("nan")

    if gap_flip == gap_flip:
        r_aug_flip = 1.0 / (1.0 + math.exp(-sigmoid_k * gap_flip))
    else:
        r_aug_flip = float("nan")

    z_orig_x = float(orig_label_logits.get(y_orig, float("nan")))
    gap_noflip = float(z_orig_x - z_orig_hat) if (z_orig_x == z_orig_x and z_orig_hat == z_orig_hat) else float("nan")

    if gap_noflip == gap_noflip:
        r_aug_noflip = 1.0 / (1.0 + math.exp(-sigmoid_k * gap_noflip))
    else:
        r_aug_noflip = float("nan")

    return {
        "reward_flip": r_flip,
        "reward_aug_flip": r_aug_flip,
        "logit_gap_abs_flip": gap_flip,       # signed gap (name kept for backward compatibility)
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
    """
    - Do NOT run inference on original x.
    - Read y_orig from file: ex['orig_pred_text'][lang].
    - Read orig_label_logits from file: ex['orig_pred_candidate_logits'][lang][label]['token_logits'][0].
    - Only run generate() for each counterfactual, compute step-0 label logits, then rewards.
    """
    # 1) original label text from file
    y_orig = ex.get("orig_pred_text", {}).get(lang, None)
    if y_orig is None:
        raise KeyError(f"Missing ex['orig_pred_text'][{lang}]")
    y_orig = str(y_orig).strip().lower()

    # 2) original label logits (FIRST token only) from file
    orig_label_logits_first: Dict[str, float] = {}
    orig_cands = ex.get("orig_pred_candidate_logits", {}).get(lang, {})
    for lb, info in orig_cands.items():
        tl = info.get("token_logits", None)
        if isinstance(tl, list) and len(tl) > 0:
            orig_label_logits_first[str(lb).strip().lower()] = float(tl[0])

    # 3) counterfactuals
    cfs = ex.get("counterfactual", {}).get(lang, [])
    if not isinstance(cfs, list):
        return ex

    hypothesis = None
    if task == "xnli":
        hypothesis = ex["hypothesis"][lang]

    for cf in cfs:
        if task == "sib200":
            prompt_cf = build_prompt_sib200(lang, cf["text"])
        else:
            prompt_cf = build_prompt_xnli(lang, cf["text"], hypothesis)

        cf_out = predict_with_confidence(
            model=model,
            tokenizer=tokenizer,
            prompt_text=prompt_cf,
            task=task,
            use_chat_template=use_chat_template,
            max_new_tokens=8,
            temperature=0.0,
        )

        cf["pred"] = cf_out["pred"]
        cf["pred_text"] = cf_out["pred_text"]
        cf["confidence"] = cf_out["confidence"]
        cf["label_logits"] = cf_out["label_logits"]
        cf["label_probs"] = cf_out["label_probs"]
        cf["score_source"] = cf_out["score_source"]
        cf["pred_token_logits"] = cf_out["pred_token_logits"]
        cf["pred_token_logprobs"] = cf_out["pred_token_logprobs"]

        rr = compute_rewards_flip_aug(
            pred=str(cf_out["pred"]).strip().lower(),
            y_orig=str(y_orig).strip().lower(),
            label_logits={k.lower(): v for k, v in cf_out["label_logits"].items()},
            orig_label_logits={k.lower(): v for k, v in orig_label_logits_first.items()},
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

    print(json.dumps(rows[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

