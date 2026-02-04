import argparse
from typing import Dict, List, Tuple, Optional

import math
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

loss_fn = nn.CrossEntropyLoss(reduction="none")

LANG2NLLB = {
    "en": "eng_Latn",
    "ar": "arb_Arab",
    "de": "deu_Latn",
    "ru": "rus_Cyrl",
    "sw": "swh_Latn",
    "vi": "vie_Latn",
    "zh": "zho_Hans",
}


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def nllb_ce_loss_and_reward(
    model,
    tokenizer,
    src_texts: List[str],
    tgt_texts: List[str],
    src_lang: str,
    tgt_lang: str = "eng_Latn",
    max_length: int = 512,
) -> Tuple[List[float], List[float]]:
    assert len(src_texts) == len(tgt_texts)

    tokenizer.src_lang = src_lang
    x = tokenizer(
        src_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(model.device)

    tokenizer.src_lang = tgt_lang
    y = tokenizer(
        tgt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    ).to(model.device)

    labels = y.input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100

    with torch.no_grad():
        out = model(**x, labels=labels)
        logits = out.logits  # [B, T, V]

    losses: List[float] = []
    rewards: List[float] = []
    for i in range(logits.size(0)):
        li = logits[i].view(-1, logits.size(-1))
        yi = labels[i].view(-1)

        # CrossEntropyLoss can't take -100 with reduction="none" safely here; mask manually
        # We compute per-token CE then mask out -100 positions.
        token_losses = loss_fn(li, yi)
        mask = (yi != -100).float()
        token_losses = token_losses * mask
        denom = float(mask.sum().item()) if float(mask.sum().item()) > 0 else 1.0
        ce = float(token_losses.sum().item() / denom)

        losses.append(ce)
        rewards.append(float(sigmoid(-ce)))   # NEW: sigmoid(-ce)

    return losses, rewards


def load_nllb(
    model_name: str = "/root/autodl-tmp/model/nllb-200-distilled-1.3B",
    device_map: str = "auto",
    dtype: str = "bfloat16",
):
    if dtype == "float16":
        torch_dtype = torch.float16
    elif dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = torch.float32

    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map=device_map, torch_dtype=torch_dtype)
    return mdl, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_lang", type=str, required=True, help="one of en/ar/de/ru/sw/vi/zh")
    ap.add_argument("--src_text", type=str, required=True)
    ap.add_argument("--tgt_text", type=str, required=True, help="English anchor text")
    ap.add_argument("--model_name", type=str, default="/root/autodl-tmp/model/nllb-200-distilled-1.3B")
    ap.add_argument("--device_map", type=str, default="auto")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    args = ap.parse_args()

    model, tokenizer = load_nllb(args.model_name, args.device_map, args.dtype)
    src = LANG2NLLB[args.src_lang]
    losses, rewards = nllb_ce_loss_and_reward(
        model, tokenizer, [args.src_text], [args.tgt_text],
        src_lang=src, tgt_lang="eng_Latn"
    )
    print({"ce_loss": losses[0], "reward_align": rewards[0]})


if __name__ == "__main__":
    main()
