import argparse
import datetime as _datetime
import json
import os
import random
from pathlib import Path
import re
import yaml
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# LoRA / PEFT
from peft import PeftModel  # pip install peft


datetime = _datetime.datetime



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

# NEW: taxi1500 labels
TAXI1500_LABEL2ID = {
    "recommendation": 0,
    "faith": 1,
    "violence": 2,
    "grace": 3,
    "sin": 4,
    "description": 5,
}
TAXI1500_ID2LABEL = {v: k for k, v in TAXI1500_LABEL2ID.items()}  # <-- NEW



def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)



def write_jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")



def safe_int(x):
    if isinstance(x, int):
        return x
    if isinstance(x, str) and x.strip().isdigit():
        return int(x.strip())
    return int(x)



def keep_last_edit_block(text: str, keep_tags: bool = True) -> str:
    matches = re.findall(r"<edit>(.*?)</edit>", text, flags=re.DOTALL)
    if not matches:
        return text.strip()
    last_inner = matches[-1].strip()
    return f"<edit>{last_inner}</edit>" if keep_tags else last_inner  # 小修：保留标签



def xnli_id2label(i: int) -> str:
    mapping = {0: "entailment", 1: "neutral", 2: "contradiction"}
    return mapping[int(i)]



def sib200_id2label(i: int) -> str:
    return SIB200_ID2LABEL[int(i)]

# NEW: taxi1500 id→label
def taxi1500_id2label(i: int) -> str:
    return TAXI1500_ID2LABEL[int(i)]  # <-- NEW



def get_prompt_template(first, second, prediction_label, target_label, language, dataset_name):
    language_dict = {
        "en": "English",
        "de": "German",
        "ar": "Arabic",
        "sw": "swahili",
        "vi": "Vietnamese",
        "zh": "Chinese",
        "ru": "Russian",
        "bg": "Bulgarian",
        "el": "Greek",
        "es": "Spanish",
        "hi": "Hindi",
        "th": "Thai",
        "tr": "Turkish",
    }

    ds = dataset_name.lower()

    if ds == "xnli":
        prompt_template = f"""
Given two sentences (premise and hypothesis) in {language_dict[language]} and their original relationship, determine whether they
entail, contradict, or are neutral to each other. Change the premise to achieve
the target relation from the original one and output the edited premise surrounding by <edit>[premise]</edit> in {language}. 


#####Begin Example####


Original relation: **entailment**
premise: A woman is talking to a man. 
hypothesis: Brown-haired woman talking to man with backpack.
Target relation: **neutral**


Step 1: Identify phrases, words in the premise leading to the entailment relation:
'man',
Step 2: Change these phrases, words to get neutral relation:
'man' to 'student'.
Step 3: replace the phrases, words from step 1 in the original text by the phrases, words, sentences
in step 2:


Edited premise: <edit>A woman is talking to a student.</edit>


#####End Example####


Request: Given two sentences (premise and hypothesis) in {language_dict[language]} and their original relationship, determine
whether they entail, contradict, or are neutral to each other. Change the premise
to achieve the neutral relation from the original one  and output the edited premise surrounding by 
<edit>[premise]</edit> in {language_dict[language]}. 


Original relation: **{prediction_label}**
premise: {first}
hypothesis: {second}
Target relation: **{target_label}**
Edited premise: 
        """
    elif ds == "sib200":
        prompt_template = f"""
Given a sentence in {language_dict[language]} classified as belonging to one of the topics: 
"science/technology", "travel", "politics", "sports", "health", "entertainment", "geography". Modify the 
sentence to change its topic to the specified target topic and output the edited sentence surrounding by 
<edit>[sentence]</edit> in {language_dict[language]}.


#####Begin Example####


Original topic: **sports**
Sentence: The athlete set a new record in the marathon.
Target topic: **health**


Step 1: Identify key phrases or words determining the original topic:
'athlete', 'record', 'marathon'.
Step 2: Modify these key phrases or words to reflect the target topic (health):
'athlete' to 'patient', 'set a new record' to 'showed improvement', 'marathon' to 'rehabilitation'.
Step 3: Replace the identified words or phrases in the original sentence:


Edited sentence: <edit>The patient showed improvement in the rehabilitation.</edit>


#####End Example####


Request: Given a sentence in {language_dict[language]} classified as belonging to one of the topics: 
"science/technology", "travel", "politics", "sports", "health", "entertainment", "geography". Modify the 
sentence to change its topic to the specified target topic and output the edited sentence surrounding by 
<edit>[sentence]</edit> in {language_dict[language]}.


Original topic: **{prediction_label}**
Sentence: {first}
Target topic: **{target_label}**
Edited sentence: 
        """
    else:  # taxi1500  <-- NEW
        prompt_template = f"""
Given a verse/text in {language_dict[language]} classified as belonging to one of the labels:
"recommendation", "faith", "violence", "grace", "sin", "description".
Modify the text to change its label to the specified target label and output ONLY the edited text surrounded by
<edit>[text]</edit> in {language_dict[language]}.


#####Begin Example####


Original label: **faith**
Text: Trust in God and do not be afraid.
Target label: **recommendation**


Step 1: Identify key phrases or words determining the original label:
'Trust in God', 'do not be afraid'
Step 2: Modify these phrases to reflect the target label (recommendation):
Replace with advice/actionable guidance.
Step 3: Replace the identified parts in the original text.


Edited text: <edit>Be kind to others and forgive those who wrong you.</edit>


#####End Example####


Request: Modify the following text in {language_dict[language]} to achieve the target label from the original one.


Original label: **{prediction_label}**
Text: {first}
Target label: **{target_label}**
Edited text:
        """

    return prompt_template



def get_parser():
    parser = argparse.ArgumentParser(description="Generate multilingual counterfactual examples from jsonl files.")


    parser.add_argument("--num_generations", type=int, default=1, help="How many outputs per input.")


    parser.add_argument("--pred_root", type=str, default="./data_qwen_pred_new", help="Input root dir.")
    parser.add_argument("--out_root", type=str, default="./data_qwen8_macro_sample_no_edit_no_min", help="Output root dir.")
    parser.add_argument("--dataset_name", type=str, default="sib200",
                        choices=["sib200", "xnli", "XNLI", "taxi1500"],  # <-- NEW
                        help="Dataset.")
    parser.add_argument("--language", type=str, default="de", help="Language (file name like de.jsonl).")


    parser.add_argument("--split", type=str, default="test", help="Only train is intended; keep as a knob for safety.")
    parser.add_argument("--max_samples", type=int, default=None, help="Process at most N lines (debug/ablation).")


    parser.add_argument("--model_name_or_path", type=str, default="google/gemma-3-4b-it", help="Base model path.")
    parser.add_argument("--hf_token", type=str, default=os.environ.get("HF_TOKEN_PATH", None), help="HF token if needed.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Cache dir.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Generation length.")
    parser.add_argument("--save_every", type=int, default=1, help="Write partial results every N samples.")


    # NEW: LoRA adapter
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="LoRA adapter directory path (contains adapter_config.json + adapter_model.safetensors). If None, run base model.",
    )
    parser.add_argument(
        "--adapter_name",
        type=str,
        default="default",
        help="Adapter name to register/use inside PEFT (default: default).",
    )
    parser.add_argument(
        "--merge_lora",
        action="store_true",
        help="(Optional) Merge LoRA into base weights for inference. Usually not necessary.",
    )


    return parser



def load_model_and_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        token=args.hf_token,
        cache_dir=args.cache_dir,
        trust_remote_code=True,
    )


    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="balanced",
        trust_remote_code=True,
        revision="main",
        token=args.hf_token,
        cache_dir=args.cache_dir,
        torch_dtype=torch.bfloat16,
    )


    if args.adapter_path:
        adapter_dir = Path(args.adapter_path)
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Adapter path not found: {adapter_dir}")


        model = PeftModel.from_pretrained(
            base_model,
            str(adapter_dir),
            adapter_name=args.adapter_name,
            is_trainable=False,
        )


        if hasattr(model, "set_adapter"):
            model.set_adapter(args.adapter_name)


        if args.merge_lora and hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()


        return model, tokenizer


    return base_model, tokenizer



def run(args):
    random.seed(args.seed)


    dataset = args.dataset_name.lower()
    split = args.split
    lang = args.language


    in_file = Path(args.pred_root) / dataset / split / f"{lang}.jsonl"
    if not in_file.exists():
        raise FileNotFoundError(f"Input file not found: {in_file}")


    now = datetime.now().strftime("%y%m%d_%H%M%S")
    model_tag = Path(args.model_name_or_path).name
    adapter_tag = Path(args.adapter_path).name if args.adapter_path else "base"
    run_dir = Path(args.out_root) / dataset / split
    out_file = run_dir / f"{lang}_{model_tag}_{adapter_tag}_{now}.jsonl"
    run_dir.mkdir(parents=True, exist_ok=True)


    with (run_dir / "args.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(vars(args), f, allow_unicode=True)


    model, tokenizer = load_model_and_tokenizer(args)
    model.eval()


    results = []
    total = 0


    for obj in tqdm(read_jsonl(in_file), desc=f"Processing {in_file.name}"):
        if args.max_samples is not None and total >= args.max_samples:
            break

        if dataset == "xnli":
            premise = obj["premise"][lang]
            hypothesis = obj["hypothesis"][lang]

            orig_pred_id = safe_int(obj["orig_pred"][lang])
            target_pred_id = safe_int(obj["target_pred"][lang])

            prediction_label = xnli_id2label(orig_pred_id)
            target_label = xnli_id2label(target_pred_id)

            prompt = get_prompt_template(
                first=premise,
                second=hypothesis,
                prediction_label=prediction_label,
                target_label=target_label,
                language=lang,
                dataset_name="XNLI",
            )
        elif dataset == "sib200":
            text = obj["text"][lang]

            orig_pred_id = safe_int(obj["orig_pred"][lang])
            target_pred_id = safe_int(obj["target_pred"][lang])

            prediction_label = sib200_id2label(orig_pred_id)
            target_label = sib200_id2label(target_pred_id)

            prompt = get_prompt_template(
                first=text,
                second=None,
                prediction_label=prediction_label,
                target_label=target_label,
                language=lang,
                dataset_name="sib200",
            )
        else:  # taxi1500
            text = obj["text"][lang]

            orig_pred_id = safe_int(obj["orig_pred"][lang])
            target_pred_id = safe_int(obj["target_pred"][lang])

            prediction_label = taxi1500_id2label(orig_pred_id)
            target_label = taxi1500_id2label(target_pred_id)

            prompt = get_prompt_template(
                first=text,
                second=None,
                prediction_label=prediction_label,
                target_label=target_label,
                language=lang,
                dataset_name="taxi1500",
            )

        system_instr = "You are an excellent assistant for text editing./no_think"
        messages = [{"role": "user", "content": system_instr + "\n\n" + prompt}]
        chat_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([chat_text], return_tensors="pt").to(model.device)

        with torch.inference_mode():
            generated = model.generate(
                **model_inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=1.1,
                top_p=0.99,
                top_k=100,
                num_return_sequences=args.num_generations,
                pad_token_id=model.config.pad_token_id,
            )

        input_len = model_inputs.input_ids.shape[1]
        gen_only = generated[:, input_len:]

        responses_raw = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
        responses = [keep_last_edit_block(r.strip(), keep_tags=True) for r in responses_raw]

        out_obj = dict(obj)
        out_obj["counterfactual"] = {
            lang: [{"gen_id": j, "text": r} for j, r in enumerate(responses)]
        }

        results.append(out_obj)
        total += 1

        if args.save_every and total % args.save_every == 0:
            write_jsonl(out_file, results)

    write_jsonl(out_file, results)
    print(f"Saved {len(results)} samples to: {out_file}")



if __name__ == "__main__":
    run(get_parser().parse_args())
