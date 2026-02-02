import os
import re
import json
import time
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, StoppingCriteria, StoppingCriteriaList


LANGS_ALLOWED = ["en", "ar", "de", "ru", "sw", "vi", "zh"]

# Your mapping (string -> id)
SIB200_LABEL2ID = {
    "science/technology": 0,
    "travel": 1,
    "politics": 2,
    "sports": 3,
    "health": 4,
    "entertainment": 5,
    "geography": 6,
}

# XNLI mapping: 0 entailment, 1 neutral, 2 contradiction
XNLI_LABEL_SPEC = "0=entailment, 1=neutral, 2=contradiction"

DEFAULT_LIMITS = {
    ("xnli", "train"): 1500,
    ("xnli", "test"): 400,
    ("xnli", "validation"): 500,
    
    ("sib200", "train"): 1500,
    ("sib200", "test"): 99,
    ("sib200", "validation"): 99,
}


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def extract_xnli_pair(ex: Dict[str, Any], srclang: str) -> Tuple[str, str]:
    premise = ex["premise"][srclang]

    hyp = ex["hypothesis"]
    if isinstance(hyp, dict) and "language" in hyp and "translation" in hyp:
        langs = hyp["language"]
        trans = hyp["translation"]
        idx = langs.index(srclang)
        hypothesis = trans[idx]
    else:
        hypothesis = hyp[srclang]

    return premise, hypothesis


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


_INT_RE = re.compile(r"(-?\d+)")


def parse_first_int(s: str) -> Optional[int]:
    m = _INT_RE.search(s.strip())
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


class StopOnSubstrings(StoppingCriteria):
    def __init__(self, tokenizer, stop_strings: List[str], start_len: int):
        super().__init__()
        self.tokenizer = tokenizer
        self.stop_strings = stop_strings
        self.start_len = start_len  # prompt length in tokens

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # decode only the newly generated part (cheap here since max_new_tokens small)
        gen_ids = input_ids[0, self.start_len:]
        if gen_ids.numel() == 0:
            return False
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return any(st in text for st in self.stop_strings)


def output_path(out_root: Path, task: str, split: str, lang: str) -> Path:
    return out_root / task / split / f"{lang}.jsonl"


def count_existing_lines(p: Path) -> int:
    if not p.exists():
        return 0
    with p.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def encode_prompt(
    tokenizer,
    prompt_text: str,
    use_chat_template: bool = True,
) -> Dict[str, torch.Tensor]:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt_text}]
        # gemma-it 推荐用 chat template + add_generation_prompt
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

    # put inputs on the same device as the first embedding weights (works well for single-GPU; for device_map=auto, accelerate handles generate)
    if hasattr(model, "device") and str(model.device) != "meta":
        input_ids = input_ids.to(model.device)
        if "attention_mask" in enc:
            enc["attention_mask"] = enc["attention_mask"].to(model.device)

    prompt_len = input_ids.shape[1]

    # stopping
    stopping = None
    if stop:
        stopping = StoppingCriteriaList([StopOnSubstrings(tokenizer, stop_strings=stop, start_len=prompt_len)])

    do_sample = temperature is not None and temperature > 0

    # pad_token_id safety
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

    seq = outputs.sequences[0]  # shape [prompt_len + gen_len]
    out_ids = seq[prompt_len:]
    out_tokens = out_ids.tolist()

    out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
    out_text_stripped = out_text.strip()

    # outputs.scores: tuple length = gen_len, each is [batch, vocab]
    token_logprobs: List[float] = []
    if outputs.scores is not None and len(outputs.scores) > 0:
        gen_len = min(len(outputs.scores), out_ids.numel())
        for t in range(gen_len):
            scores_t = outputs.scores[t][0]  # [vocab]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["xnli", "sib200"], required=True)
    ap.add_argument("--split", choices=["train", "test", "validation"], required=True)
    ap.add_argument("--lang", choices=LANGS_ALLOWED, required=True)

    ap.add_argument("--data_dir", type=str, default="../data")
    ap.add_argument("--out_dir", type=str, default="../data_gemma2_pred")

    # local model controls
    #ap.add_argument("--local_model_dir", type=str, default="/root/autodl-tmp/model/gemma-2-9b-it")
    ap.add_argument("--local_model_dir", type=str, default="/root/autodl-tmp/model/Qwen3-8B")
    
    ap.add_argument("--device_map", type=str, default="auto")  # "cuda" / "auto"
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    ap.add_argument("--local_files_only", action="store_true", help="Do not download anything; load only from local dir.")
    ap.add_argument("--use_chat_template", action="store_true", help="Use tokenizer.apply_chat_template for -it models.")

    ap.add_argument("--max_examples", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="Resume by skipping already-written lines")
    ap.add_argument("--max_tokens", type=int, default=1)  # interpreted as max_new_tokens
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    if args.max_examples is None:
        args.max_examples = DEFAULT_LIMITS.get((args.task, args.split), 100)

    # dtype
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

    # load dataset file
    if args.task == "xnli":
        fn = {
            "train": "xnli/training_set.json",
            "test": "xnli/test_set.json",
            "validation": "xnli/validation_set.json",
        }[args.split]
        data_path = data_dir / fn
    else:
        fn = {
            "train": "sib200/sib200_train.json",
            "test": "sib200/sib200_test.json",
            "validation": "sib200/sib200_validation.json",
        }[args.split]
        data_path = data_dir / fn

    data = load_json_list(data_path)

    # init local model + tokenizer
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

                    gen = call_sample_and_logprob_local(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_text=prompt_text,
                        max_new_tokens=args.max_tokens,
                        temperature=args.temperature,
                        stop=None,
                        use_chat_template=args.use_chat_template,
                    )
                    pred = parse_first_int(gen["text"])

                    out_obj = {
                        "index": ex.get("index", i),
                        "premise": {args.lang: premise},
                        "hypothesis": {args.lang: hypothesis},
                        "label": ex.get("label", None),
                        "orig_pred": {args.lang: pred},
                        "orig_pred_text": {args.lang: gen["text"]},
                        "orig_pred_logprob": {args.lang: gen["total_logprob"]},
                        "orig_pred_token_logprobs": {args.lang: gen["token_logprobs"]},
                    }

                else:
                    text = ex["text"][args.lang]
                    prompt_text = build_prompt_sib200(args.lang, text)

                    gen = call_sample_and_logprob_local(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_text=prompt_text,
                        max_new_tokens=args.max_tokens,
                        temperature=args.temperature,
                        stop=None,
                        use_chat_template=args.use_chat_template,
                    )
                    pred = parse_first_int(gen["text"])

                    out_obj = {
                        "index": ex.get("index", i),
                        "text": {args.lang: text},
                        "label": ex.get("label", None),
                        "orig_pred": {args.lang: pred},
                        "orig_pred_text": {args.lang: gen["text"]},
                        "orig_pred_logprob": {args.lang: gen["total_logprob"]},
                        "orig_pred_token_logprobs": {args.lang: gen["token_logprobs"]},
                    }

                wf.write(json.dumps(out_obj, ensure_ascii=False) + "\n")
                wf.flush()
                written += 1

            except Exception as e:
                err_obj = {
                    "index": ex.get("index", i),
                    "error": str(e),
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
