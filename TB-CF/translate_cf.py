#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import io
import json
import os
import re
from typing import Dict, List, Tuple, Optional

PROMPT_TEMPLATE = (
    "You are a professional translator, fluent in English and {language}. /no_think"
    "Translate the following English text to {language} accurately and naturally, "
    "preserving its tone, style, and any cultural nuances. "
    "Text to translate: {counterfactual}"
)

LANG_CODE_TO_NAME = {
    "ar": "Arabic",
    "de": "German",
    "en": "English",
    "ru": "Russian",
    "sw": "Swahili",
    "vi": "Vietnamese",
    "zh": "Chinese",
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "nl": "Dutch",
    "tr": "Turkish",
    "pl": "Polish",
    "uk": "Ukrainian",
    "hi": "Hindi",
    "id": "Indonesian",
    "ja": "Japanese",
    "ko": "Korean",
    "th": "Thai",
}


def read_jsonl(path: str) -> List[dict]:
    data = []
    with io.open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSON decode error in {path}:{line_no}: {e}")
    return data


def write_jsonl(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def detect_lang_from_filename(fname: str) -> str:
    base = os.path.basename(fname)
    # Expect: "{lang}_....jsonl"
    return base.split("_", 1)[0]


def find_en_file(split_dir: str, en_glob: str = "en_*.jsonl") -> str:
    candidates = sorted(glob.glob(os.path.join(split_dir, en_glob)))
    if not candidates:
        raise FileNotFoundError(f"No English jsonl found in {split_dir} with glob={en_glob}")
    if len(candidates) > 1:
        # pick the first deterministically, but warn via stderr-like print
        print(f"[WARN] Multiple English files found, pick first: {candidates[0]}")
    return candidates[0]


def extract_en_cf0_map(en_rows: List[dict]) -> Dict[int, str]:
    m = {}
    for r in en_rows:
        idx = r.get("index", None)
        if idx is None:
            continue
        cf = (r.get("counterfactual") or {}).get("en")
        if not cf:
            continue
        cf0 = None
        for item in cf:
            if item.get("gen_id") == 0:
                cf0 = item.get("text", "")
                break
        if cf0 is None:
            continue
        m[int(idx)] = cf0
    return m


def clean_translation(text: str) -> str:
    t = (text or "").strip()
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.DOTALL).strip()
    t = re.sub(r'^\s*(Translation|translation|译文|翻译)\s*[:：]\s*', '', t).strip()
    if (t.startswith('"') and t.endswith('"')) or (t.startswith("“") and t.endswith("”")):
        t = t[1:-1].strip()
    return t



class TranslatorBase:
    def translate_batch(self, prompts: List[str]) -> List[str]:
        raise NotImplementedError


class VLLMTranslator(TranslatorBase):
    def __init__(
        self,
        model: str,
        dtype: str = "bfloat16",
        tensor_parallel_size: int = 1,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        gpu_memory_utilization: float = 0.90,
        trust_remote_code: bool = True,
    ):
        from vllm import LLM, SamplingParams

        self.SamplingParams = SamplingParams
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        self.llm = LLM(
            model=model,
            dtype=dtype,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
        )

    def translate_batch(self, prompts: List[str]) -> List[str]:
        outputs = self.llm.generate(prompts, self.sampling_params)
        # Keep order consistent with prompts
        # vLLM returns in the same order as input prompts in typical usage
        res = []
        for out in outputs:
            res.append(clean_translation(out.outputs[0].text))
        return res


class HFTranslator(TranslatorBase):
    def __init__(
        self,
        model: str,
        max_new_tokens: int = 256,
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
        trust_remote_code: bool = True,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
            padding_side="left",
            trust_remote_code=trust_remote_code,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        dtype = getattr(torch, torch_dtype) if hasattr(torch, torch_dtype) else torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        self.max_new_tokens = max_new_tokens

    def _format_prompt(self, raw_prompt: str) -> str:
        # Prefer chat template if available
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            messages = [
                {"role": "user", "content": raw_prompt},
            ]
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        return raw_prompt

    def translate_batch(self, prompts: List[str]) -> List[str]:
        formatted = [self._format_prompt(p) for p in prompts]
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        with self.torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
            )

        in_len = inputs["input_ids"].shape[1]
        gen_ids = out_ids[:, in_len:]
        texts = self.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        return [clean_translation(t) for t in texts]


def chunked(lst: List, n: int):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def parse_list_arg(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    ap.add_argument("--translator_model", type=str, required=True)

    ap.add_argument("--input_root", type=str, required=True)
    ap.add_argument("--output_root", type=str, required=True)
    ap.add_argument("--datasets", type=str, default="all", help="e.g. sib200,taxi1500,xnli or all")
    ap.add_argument("--split", type=str, default="test")

    ap.add_argument("--languages", type=str, default="all", help="e.g. de,ru,zh or all")
    ap.add_argument("--en_glob", type=str, default="en_*.jsonl")

    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=256)

    # vLLM options
    ap.add_argument("--tp", type=int, default=1, help="tensor_parallel_size for vLLM")
    ap.add_argument("--dtype", type=str, default="bfloat16")
    ap.add_argument("--gpu_mem_util", type=float, default=0.90)

    # HF options
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--torch_dtype", type=str, default="bfloat16")

    ap.add_argument("--trust_remote_code", action="store_true", default=True)
    args = ap.parse_args()

    datasets = []
    if args.datasets == "all":
        datasets = sorted([d for d in os.listdir(args.input_root) if os.path.isdir(os.path.join(args.input_root, d))])
    else:
        datasets = parse_list_arg(args.datasets)

    if args.languages == "all":
        target_langs = None  # means "discover from files"
    else:
        target_langs = set(parse_list_arg(args.languages))

    if args.backend == "vllm":
        translator: TranslatorBase = VLLMTranslator(
            model=args.translator_model,
            dtype=args.dtype,
            tensor_parallel_size=args.tp,
            max_tokens=args.max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            gpu_memory_utilization=args.gpu_mem_util,
            trust_remote_code=args.trust_remote_code,
        )
    else:
        translator = HFTranslator(
            model=args.translator_model,
            max_new_tokens=args.max_new_tokens,
            device_map=args.device_map,
            torch_dtype=args.torch_dtype,
            trust_remote_code=args.trust_remote_code,
        )

    cache: Dict[Tuple[str, str], str] = {}

    for ds in datasets:
        split_dir = os.path.join(args.input_root, ds, args.split)
        if not os.path.isdir(split_dir):
            print(f"[WARN] Skip missing dir: {split_dir}")
            continue

        en_path = find_en_file(split_dir, args.en_glob)
        en_rows = read_jsonl(en_path)
        en_cf0_map = extract_en_cf0_map(en_rows)
        if not en_cf0_map:
            print(f"[WARN] Empty en cf0 map in {en_path}, skip dataset={ds}")
            continue

        lang_files = sorted(glob.glob(os.path.join(split_dir, "*.jsonl")))
        for lf in lang_files:
            lang = detect_lang_from_filename(lf)
            if lang == "en":
                continue
            if target_langs is not None and lang not in target_langs:
                continue

            rows = read_jsonl(lf)

            # build prompts (only where we can find matching en cf0)
            prompts: List[str] = []
            row_indices: List[int] = []

            lang_name = LANG_CODE_TO_NAME.get(lang, lang)

            for i, r in enumerate(rows):
                idx = r.get("index", None)
                if idx is None:
                    continue
                idx = int(idx)
                src_cf = en_cf0_map.get(idx, None)
                if not src_cf:
                    continue

                key = (lang, src_cf)
                if key in cache:
                    continue

                prompt = PROMPT_TEMPLATE.format(language=lang_name, counterfactual=src_cf)
                prompts.append(prompt)
                row_indices.append(idx)

            # translate in batches and fill cache
            for batch in chunked(prompts, max(1, args.batch_size)):
                translations = translator.translate_batch(batch)
                for p, t in zip(batch, translations):
                    # recover src_cf from prompt (last "Text to translate: ...")
                    # safer: parse by split
                    m = p.split("Text to translate:", 1)
                    src_cf = m[1].strip() if len(m) == 2 else p
                    cache[(lang, src_cf)] = t

            # write updated rows
            updated = 0
            for r in rows:
                idx = r.get("index", None)
                if idx is None:
                    continue
                idx = int(idx)
                src_cf = en_cf0_map.get(idx, None)
                if not src_cf:
                    continue

                trans = cache.get((lang, src_cf), None)
                if trans is None or not trans.strip():
                    continue

                r["counterfactual"] = {lang: [{"gen_id": 0, "text": trans}]}
                updated += 1

            out_dir = os.path.join(args.output_root, ds, args.split)
            out_path = os.path.join(out_dir, os.path.basename(lf))
            write_jsonl(out_path, rows)
            print(f"[OK] dataset={ds} lang={lang} updated={updated} -> {out_path}")


if __name__ == "__main__":
    main()
