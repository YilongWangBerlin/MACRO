import argparse
import json
import random
import re
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm.auto import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Gemma3ForConditionalGeneration,
)
from peft import PeftConfig, PeftModel


LETTERS = list("ABCDEFGHIJ")
SCRIPT_DIR = Path(__file__).resolve().parent

SYSTEM_PROMPT = (
    "You are a careful multiple-choice evaluator.\n"
    "Read the question and the provided options carefully.\n"
    "You may think briefly, but you must choose exactly one option.\n"
    "Your final line MUST be exactly:\n"
    "<final_answer>X</final_answer>\n"
    "where X is exactly one valid option letter.\n"
    "Do not output multiple final answers."
)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate one JSON file of MCQ data with a local/base/LoRA model.")

    parser.add_argument("--input-json", type=str, required=True, help="Path to one input JSON file.")
    parser.add_argument("--dataset-type", type=str, default="auto", choices=["auto", "mmlu", "mmlu_prox"])

    parser.add_argument("--model-name-or-path", type=str, default=None, help="HF repo or local base model path.")
    parser.add_argument("--adapter-path", type=str, default=None, help="Optional PEFT/LoRA adapter path.")
    parser.add_argument("--model-tag", type=str, default=None, help="Optional output folder name override.")

    parser.add_argument("--output-root", type=str, default=None, help="Default: general/llm_eval/outputs")

    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-input-length", type=int, default=4096)

    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)

    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--device-map", type=str, default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)

    parser.add_argument("--resume", action="store_true", help="Skip already written rows in output jsonl.")
    parser.add_argument("--overwrite", action="store_true", help="Delete existing output file first.")
    parser.add_argument("--save-prompt", action="store_true", help="Store the final rendered prompt in each jsonl row.")

    return parser.parse_args()


def sanitize_name(x: str) -> str:
    x = x.strip().replace("\\", "/").rstrip("/")
    x = x.split("/")[-1]
    x = re.sub(r"[^a-zA-Z0-9._-]+", "_", x)
    return x


def get_torch_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[dtype_name]


def resolve_base_model_name(model_name_or_path: Optional[str], adapter_path: Optional[str]) -> str:
    if model_name_or_path:
        return model_name_or_path
    if not adapter_path:
        raise ValueError("You must provide either --model-name-or-path or --adapter-path.")
    peft_cfg = PeftConfig.from_pretrained(adapter_path)
    if not peft_cfg.base_model_name_or_path:
        raise ValueError("Cannot infer base model from adapter_config.json in the adapter path.")
    return peft_cfg.base_model_name_or_path


def is_gemma3_model(model_name: str) -> bool:
    return "gemma-3" in model_name.lower()


def has_chat_template(tokenizer) -> bool:
    return getattr(tokenizer, "chat_template", None) is not None


def resolve_output_root(output_root: Optional[str]) -> Path:
    if output_root is None:
        return SCRIPT_DIR / "outputs"
    return Path(output_root)


def load_rows(input_json: str) -> List[Dict[str, Any]]:
    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON must contain a list of objects.")
    return data


def infer_dataset_type(input_path: Path, rows: List[Dict[str, Any]], user_choice: str) -> str:
    if user_choice != "auto":
        return user_choice

    parts = [p.lower() for p in input_path.parts]
    if "mmlu_prox" in parts:
        return "mmlu_prox"
    if "mmlu" in parts:
        return "mmlu"

    if rows:
        row = rows[0]
        if "choices" in row:
            return "mmlu"
        if any(re.fullmatch(r"option_\d+", k) for k in row.keys()):
            return "mmlu_prox"

    raise ValueError("Cannot infer dataset type. Please pass --dataset-type mmlu or --dataset-type mmlu_prox.")


def extract_options(row: Dict[str, Any], dataset_type: str) -> List[str]:
    if dataset_type == "mmlu":
        if "choices" not in row or not isinstance(row["choices"], list):
            raise ValueError("MMLU row must contain a list field `choices`.")
        return [str(x) for x in row["choices"]]

    if dataset_type == "mmlu_prox":
        option_pairs = []
        for k, v in row.items():
            m = re.fullmatch(r"option_(\d+)", k)
            if m:
                idx = int(m.group(1))
                if v is None:
                    continue
                v = str(v).strip()
                if v == "":
                    continue
                option_pairs.append((idx, v))

        option_pairs = sorted(option_pairs, key=lambda x: x[0])
        if not option_pairs:
            raise ValueError("MMLU-ProX row has no valid options.")
        return [v for _, v in option_pairs]

    raise ValueError(f"Unsupported dataset_type: {dataset_type}")


def extract_gold_index(row: Dict[str, Any], dataset_type: str) -> Optional[int]:
    if dataset_type == "mmlu":
        if "answer" not in row:
            return None
        try:
            return int(row["answer"])
        except Exception:
            return None

    if dataset_type == "mmlu_prox":
        if "answer_index" not in row:
            return None
        try:
            return int(row["answer_index"])
        except Exception:
            return None

    return None


def build_question_block(row: Dict[str, Any], dataset_type: str) -> str:
    question = str(row.get("question", "")).strip()
    if not question:
        raise ValueError("Missing `question` field.")

    options = extract_options(row, dataset_type)
    letters = LETTERS[: len(options)]
    option_text = "\n".join(f"{letters[i]}. {options[i]}" for i in range(len(options)))

    meta_lines = []
    if dataset_type == "mmlu":
        if "subject" in row:
            meta_lines.append(f"Subject: {row['subject']}")
    elif dataset_type == "mmlu_prox":
        if "category" in row:
            meta_lines.append(f"Category: {row['category']}")
        if "src" in row:
            meta_lines.append(f"Source: {row['src']}")

    meta_prefix = ""
    if meta_lines:
        meta_prefix = "\n".join(meta_lines) + "\n\n"

    return (
        f"{meta_prefix}"
        f"Question:\n{question}\n\n"
        f"Options:\n{option_text}\n\n"
        f"Instructions:\n"
        f"- Choose exactly one option.\n"
        f"- Use only the provided options.\n"
        f"- Your final line must be exactly: <final_answer>X</final_answer>\n"
        f"- X must be one of: {', '.join(letters)}\n"
    )


def render_prompt(user_prompt: str, tokenizer, processor, gemma3: bool) -> str:
    if gemma3 and processor is not None:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}]},
        ]
        try:
            return processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    if has_chat_template(tokenizer):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass

    return f"{SYSTEM_PROMPT}\n\n{user_prompt}\nAssistant:\n"


def parse_final_answer(text: str, num_options: int) -> Tuple[Optional[str], Optional[int], str]:
    valid_letters = LETTERS[:num_options]
    valid_set = set(valid_letters)

    patterns = [
        r"<final_answer>\s*([A-Ja-j])\s*</final_answer>",
        r"<answer>\s*([A-Ja-j])\s*</answer>",
        r"final[_ ]?answer\s*[:：]?\s*([A-Ja-j])",
        r"answer\s*[:：]?\s*([A-Ja-j])",
        r"option\s*[:：]?\s*([A-Ja-j])",
    ]

    for p in patterns:
        m = re.search(p, text, flags=re.IGNORECASE)
        if m:
            letter = m.group(1).upper()
            if letter in valid_set:
                return letter, valid_letters.index(letter), "tag_or_keyword_match"

    standalone = re.findall(r"\b([A-Ja-j])\b", text)
    standalone = [x.upper() for x in standalone if x.upper() in valid_set]
    if standalone:
        letter = standalone[-1]
        return letter, valid_letters.index(letter), "fallback_last_letter"

    return None, None, "parse_failed"


def make_output_dir(
    output_root: Path,
    model_tag: Optional[str],
    base_model_name: str,
    adapter_path: Optional[str],
    dataset_type: str,
) -> Path:
    if model_tag:
        final_model_tag = model_tag
    else:
        base_tag = sanitize_name(base_model_name)
        if adapter_path:
            final_model_tag = f"{base_tag}__adapter__{sanitize_name(adapter_path)}"
        else:
            final_model_tag = f"{base_tag}__base"

    output_dir = output_root / final_model_tag / dataset_type
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def read_processed_indices(output_path: Path) -> set:
    done = set()
    if not output_path.exists():
        return done

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "__source_idx" in obj:
                    done.add(int(obj["__source_idx"]))
            except Exception:
                continue
    return done


def batched(items: List[Any], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def get_input_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_tokenizer_processor(args):
    base_model_name = resolve_base_model_name(args.model_name_or_path, args.adapter_path)
    gemma3 = is_gemma3_model(base_model_name)

    common_kwargs = {
        "device_map": args.device_map,
        "trust_remote_code": args.trust_remote_code,
    }
    dtype = get_torch_dtype(args.dtype)
    if dtype != "auto":
        common_kwargs["torch_dtype"] = dtype

    processor = None

    if gemma3:
        processor = AutoProcessor.from_pretrained(
            base_model_name,
            trust_remote_code=args.trust_remote_code,
        )
        tokenizer = processor.tokenizer
        model = Gemma3ForConditionalGeneration.from_pretrained(
            base_model_name,
            **common_kwargs,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            trust_remote_code=args.trust_remote_code,
            use_fast=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            **common_kwargs,
        )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
            model.resize_token_embeddings(len(tokenizer))

    tokenizer.padding_side = "left"

    if args.adapter_path:
        model = PeftModel.from_pretrained(
            model,
            args.adapter_path,
            is_trainable=False,
        )

    model.eval()
    return model, tokenizer, processor, base_model_name, gemma3


def main():
    args = parse_args()
    set_seed(args.seed)

    input_json = Path(args.input_json)
    rows = load_rows(str(input_json))
    dataset_type = infer_dataset_type(input_json, rows, args.dataset_type)

    start_index = args.start_index
    end_index = len(rows) if args.end_index is None else min(args.end_index, len(rows))
    rows = rows[start_index:end_index]

    if args.max_samples is not None:
        rows = rows[: args.max_samples]

    output_root = resolve_output_root(args.output_root)

    model, tokenizer, processor, base_model_name, gemma3 = load_model_tokenizer_processor(args)

    output_dir = make_output_dir(
        output_root=output_root,
        model_tag=args.model_tag,
        base_model_name=base_model_name,
        adapter_path=args.adapter_path,
        dataset_type=dataset_type,
    )
    output_path = output_dir / f"{input_json.stem}.jsonl"

    if output_path.exists() and args.overwrite and not args.resume:
        output_path.unlink()

    if output_path.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --resume or --overwrite.")

    processed_indices = read_processed_indices(output_path) if args.resume else set()

    pending = []
    for local_i, row in enumerate(rows):
        global_idx = start_index + local_i
        if global_idx in processed_indices:
            continue
        pending.append((global_idx, row))

    if not pending:
        print(f"Nothing to do. Output exists and all rows are done: {output_path}")
        return

    with open(output_path, "a", encoding="utf-8") as fout:
        for batch in tqdm(list(batched(pending, args.batch_size)), desc="Evaluating"):
            batch_indices = [x[0] for x in batch]
            batch_rows = [x[1] for x in batch]

            prompts = []
            batch_option_lists = []

            for row in batch_rows:
                options = extract_options(row, dataset_type)
                batch_option_lists.append(options)
                prompt = render_prompt(
                    build_question_block(row, dataset_type),
                    tokenizer=tokenizer,
                    processor=processor,
                    gemma3=gemma3,
                )
                prompts.append(prompt)

            tokenized = tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_input_length,
            )

            device = get_input_device(model)
            tokenized = {k: v.to(device) for k, v in tokenized.items()}

            gen_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "pad_token_id": tokenizer.pad_token_id,
                "use_cache": True,
                "do_sample": args.temperature > 0,
            }

            if tokenizer.eos_token_id is not None:
                gen_kwargs["eos_token_id"] = tokenizer.eos_token_id

            if args.temperature > 0:
                gen_kwargs["temperature"] = args.temperature
                gen_kwargs["top_p"] = args.top_p

            with torch.inference_mode():
                outputs = model.generate(**tokenized, **gen_kwargs)

            prompt_len = tokenized["input_ids"].shape[1]
            generated_ids = outputs[:, prompt_len:]
            decoded = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            for src_idx, row, prompt, raw_text, options in zip(batch_indices, batch_rows, prompts, decoded, batch_option_lists):
                pred_letter, pred_index, parse_status = parse_final_answer(raw_text, len(options))
                pred_text = options[pred_index] if pred_index is not None and pred_index < len(options) else None
                gold_index = extract_gold_index(row, dataset_type)

                out_row = deepcopy(row)
                out_row["__source_idx"] = src_idx
                out_row["__dataset_type"] = dataset_type
                out_row["__input_json"] = str(input_json)
                out_row["__resolved_base_model"] = base_model_name
                out_row["__model_name_or_path"] = args.model_name_or_path
                out_row["__adapter_path"] = args.adapter_path
                out_row["__pred_letter"] = pred_letter
                out_row["__pred_index"] = pred_index
                out_row["__pred_text"] = pred_text
                out_row["__num_options"] = len(options)
                out_row["__raw_generation"] = raw_text
                out_row["__parse_status"] = parse_status
                out_row["__gold_index"] = gold_index
                out_row["__is_correct"] = (pred_index == gold_index) if (pred_index is not None and gold_index is not None) else None
                out_row["__timestamp"] = int(time.time())

                if args.save_prompt:
                    out_row["__prompt"] = prompt

                fout.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                fout.flush()

    print(f"Done. Results written to: {output_path}")


if __name__ == "__main__":
    main()