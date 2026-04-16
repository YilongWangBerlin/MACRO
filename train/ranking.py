#!/usr/bin/env python3
"""
Build DPO pairs by choosing max-total-reward vs min-total-reward among CFEs.

New additions:
- reward_flip is recomputed: 1 iff cf_pred_id == target_pred_id
- reward_soft_flip is added: 1 iff cf_pred_id != orig_pred_id
- reward_soft_flip can be included in total reward via --reward_components soft_flip
- Pair selection:
  1) chosen = max total_reward CFE
     rejected = min total_reward CFE among those with reward_soft_flip==1 (must exist)
  2) If any deduped CFE text == original (after strip):
       rejected = explicitly constructed original sentence
       chosen   = highest-total_reward CFE whose text != original (if none, skip)
  3) If chosen_text == rejected_text after strip: SKIP

Also:
- Optionally rewrite reward jsonl under --reward_root with recomputed reward_flip and
  new reward_soft_flip (default writes a new file with suffix to avoid overwriting).
- align skip language is parameterized by --tgt_lang (default 'en').
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Tuple

import yaml

SIB200_LABEL2ID = {
    "entertainment": 0,
    "geography": 1,
    "health": 2,
    "politics": 3,
    "science/technology": 4,
    "sports": 5,
    "travel": 6,
}

XNLI_LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}

TAXI1500_LABEL2ID = {
    "recommendation": 0,
    "faith": 1,
    "violence": 2,
    "grace": 3,
    "sin": 4,
    "description": 5,
}


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(path: Path, records: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def parse_kv_list(items: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for it in items:
        if "=" not in it:
            raise ValueError(f"Bad kv item: {it}, expected like en=1000")
        k, v = it.split("=", 1)
        out[k.strip()] = int(v.strip())
    return out


def ensure_edit_tags(s: str) -> str:
    s = (s or "").strip()
    if "<edit>" in s and "</edit>" in s:
        return s
    return f"<edit>{s}</edit>"


def pick_aug_reward(cf: Dict[str, Any]) -> float:
    flip = float(cf.get("reward_flip", 0.0))
    if flip == 1.0:
        return float(cf.get("reward_aug_flip", 0.0))
    return float(cf.get("reward_aug_noflip", 0.0))


def _label_to_id(task: str, label: Any) -> int:
    if label is None:
        raise ValueError("label is None")

    if isinstance(label, (int, float)) and int(label) == label:
        return int(label)

    s = str(label).strip().lower()

    if s.isdigit() or (s.startswith("-") and s[1:].isdigit()):
        return int(s)

    if task == "sib200":
        if s not in SIB200_LABEL2ID:
            raise ValueError(f"Unknown sib200 label string: {s}")
        return int(SIB200_LABEL2ID[s])

    if task == "xnli":
        if s not in XNLI_LABEL2ID:
            raise ValueError(f"Unknown xnli label string: {s}")
        return int(XNLI_LABEL2ID[s])

    if task == "taxi1500":
        if s not in TAXI1500_LABEL2ID:
            raise ValueError(f"Unknown taxi1500 label string: {s}")
        return int(TAXI1500_LABEL2ID[s])

    raise ValueError(f"Unknown task: {task}")


def recompute_flip_rewards(
    task: str,
    cf: Dict[str, Any],
    orig_pred_id: int,
    target_pred_id: int,
) -> Tuple[float, float, int]:
    if "pred" in cf and cf["pred"] is not None:
        cf_pred_id = _label_to_id(task, cf["pred"])
    elif "pred_text" in cf and cf["pred_text"] is not None:
        cf_pred_id = _label_to_id(task, cf["pred_text"])
    else:
        raise ValueError("Counterfactual missing 'pred'/'pred_text'")

    reward_flip = 1.0 if int(cf_pred_id) == int(target_pred_id) else 0.0
    reward_soft_flip = 1.0 if int(cf_pred_id) != int(orig_pred_id) else 0.0
    return float(reward_flip), float(reward_soft_flip), int(cf_pred_id)


def compute_total_reward(
    cf: Dict[str, Any],
    lang: str,
    tgt_lang: str,
    components: List[str],
    weights: Dict[str, float],
) -> Tuple[float, Dict[str, float]]:
    """
    Weighted SUM over available components (no normalization).
    - flip: reward_flip (recomputed upstream)
    - soft_flip: reward_soft_flip (recomputed upstream)
    - edit: reward_edit
    - align: reward_align (skipped for tgt_lang)
    - aug: reward_aug_flip or reward_aug_noflip depending on reward_flip
    """
    used: Dict[str, float] = {}
    total = 0.0

    for k in components:
        w = float(weights[k])

        if k == "flip":
            if "reward_flip" not in cf:
                continue
            r = float(cf["reward_flip"])

        elif k == "soft_flip":
            if "reward_soft_flip" not in cf:
                continue
            r = float(cf["reward_soft_flip"])

        elif k == "edit":
            if "reward_edit" not in cf:
                continue
            r = float(cf["reward_edit"])

        elif k == "align":
            if lang == tgt_lang or ("reward_align" not in cf):
                continue
            r = float(cf["reward_align"])

        elif k == "aug":
            if ("reward_aug_flip" not in cf) and ("reward_aug_noflip" not in cf):
                continue
            r = float(pick_aug_reward(cf))

        else:
            raise ValueError(f"Unknown reward component: {k}")

        used[f"reward_{k}"] = r
        total += w * r

    return float(total), used


def load_prompt_fns():
    import importlib.util

    this_file = Path(__file__).resolve()
    repo_root = this_file.parent.parent
    gen_path = repo_root / "sample" / "generate_counterfactual.py"
    if not gen_path.exists():
        raise FileNotFoundError(f"Cannot find: {gen_path}")

    spec = importlib.util.spec_from_file_location("generate_counterfactual", str(gen_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create import spec for: {gen_path}")

    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    required = ["get_prompt_template", "xnli_id2label", "sib200_id2label", "taxi1500_id2label", "safe_int"]
    missing = [k for k in required if not hasattr(mod, k)]
    if missing:
        raise RuntimeError(f"{gen_path} missing symbols: {missing}")

    return mod.get_prompt_template, mod.xnli_id2label, mod.sib200_id2label, mod.taxi1500_id2label, mod.safe_int


def get_parser():
    ap = argparse.ArgumentParser("Build DPO pairs by max vs min total reward among CFEs.")

    ap.add_argument("--reward_root", type=str, default="../data_qwen_reward",
                    help="Root containing reward jsonl files.")
    ap.add_argument("--task", type=str, required=True, choices=["xnli", "sib200", "taxi1500"])
    ap.add_argument("--split", type=str, default="train", choices=["train"])
    ap.add_argument("--out_root", type=str, default=None)
    ap.add_argument("--tag", type=str, default="qwen")

    ap.add_argument("--languages", type=str, nargs="+", required=True)
    ap.add_argument("--max_per_lang", type=str, nargs="*", default=[])
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--tgt_lang", type=str, default="en",
                    help="Language for which reward_align is skipped (previously hard-coded as 'en').")

    ap.add_argument("--reward_components", type=str, nargs="+",
                    default=["flip", "soft_flip", "edit", "aug", "align"],
                    choices=["flip", "soft_flip", "edit", "aug", "align"])
    ap.add_argument("--reward_weights", type=str, nargs="+",
                    default=["flip=1.0", "soft_flip=1.0", "edit=1.0", "aug=1.0", "align=1.0"])

    ap.add_argument("--force_edit_tags", action="store_true", default=True)
    ap.add_argument("--no_force_edit_tags", action="store_true")

    ap.add_argument("--system_instr", type=str,
                    default="You are an excellent assistant for text editing./no_think")
    ap.add_argument("--prepend_system_instr", action="store_true", default=True)
    ap.add_argument("--no_prepend_system_instr", action="store_true")

    ap.add_argument("--identical_mode", type=str, default="text", choices=["text"])

    ap.add_argument("--file_glob", type=str, default="*Qwen*.jsonl",
                    help="Glob under {task}/{split} to find reward files for each language.")

    ap.add_argument("--max_cfes", type=int, default=10,
                    help="Consider at most first N CFEs per example (default 10).")

    ap.add_argument("--rewrite_reward_root", action="store_true", default=True,
                    help="Write back recomputed reward_flip and reward_soft_flip into a new jsonl under reward_root.")
    ap.add_argument("--rewrite_reward_suffix", type=str, default="with_flipsoft",
                    help="Suffix for rewritten reward jsonl filename: {lang}_{suffix}.jsonl")

    ap.add_argument("--rewrite_all_cfes", action="store_true", default=True,
                    help="Recompute flip/soft_flip for ALL CFEs when rewriting reward jsonl.")

    return ap


def run(args):
    random.seed(args.seed)

    get_prompt_template, xnli_id2label, sib200_id2label, taxi1500_id2label, safe_int = load_prompt_fns()
    max_per_lang = parse_kv_list(args.max_per_lang) if args.max_per_lang else {}

    weight_map: Dict[str, float] = {}
    for kv in args.reward_weights:
        if "=" not in kv:
            raise ValueError(f"Bad --reward_weights item: {kv}")
        k, v = kv.split("=", 1)
        weight_map[k.strip()] = float(v.strip())
    for k in args.reward_components:
        if k not in weight_map:
            raise ValueError(f"Missing weight for component '{k}', please add to --reward_weights")

    rewardtag = "+".join(args.reward_components)
    out_root = Path(args.out_root) if args.out_root else Path(f"../data_{args.tag}_dpo_triple_{rewardtag}")
    out_task_dir = out_root / args.task / args.split
    out_root.mkdir(parents=True, exist_ok=True)
    out_task_dir.mkdir(parents=True, exist_ok=True)

    force_tags = bool(args.force_edit_tags) and (not bool(args.no_force_edit_tags))
    prepend_sys = bool(args.prepend_system_instr) and (not bool(args.no_prepend_system_instr))
    system_instr = (args.system_instr or "").strip()

    summary: Dict[str, Any] = {
        "task": args.task,
        "split": args.split,
        "languages": args.languages,
        "tgt_lang": args.tgt_lang,
        "reward_components": args.reward_components,
        "reward_weights": {k: weight_map[k] for k in args.reward_components},
        "pairing": (
            "dedup_by_text_keep_max_total; rank_desc; "
            "initial chosen=max_total; rejected=min_total_among_softflip1; "
            "then override with explicit-original rejection if any exact-original text present; "
            "skip if <2 after dedup"
        ),
        "prompt_setting": {
            "prepend_system_instr": bool(prepend_sys),
            "system_instr": system_instr if prepend_sys else None,
        },
        "force_edit_tags": bool(force_tags),
        "file_glob": args.file_glob,
        "max_cfes": int(args.max_cfes),
        "flip_definition": "reward_flip = 1 iff cf_pred == target_pred",
        "soft_flip_definition": "reward_soft_flip = 1 iff cf_pred != orig_pred",
        "rewrite_reward_root": bool(args.rewrite_reward_root),
        "rewrite_reward_suffix": args.rewrite_reward_suffix,
        "rewrite_all_cfes": bool(args.rewrite_all_cfes),
        "align_skip_lang": args.tgt_lang,
    }

    lang_stats: Dict[str, Any] = {}

    for lang in args.languages:
        in_dir = Path(args.reward_root) / args.task / args.split
        cand_files = sorted(in_dir.glob(f"{lang}_{args.file_glob}"))
        if not cand_files:
            raise FileNotFoundError(
                f"No reward file for lang={lang} under {in_dir} with glob {lang}_{args.file_glob}"
            )
        in_file = cand_files[-1]

        cap = max_per_lang.get(lang, None)
        records_out: List[Dict[str, Any]] = []
        n_written = 0

        rewritten_reward_records: List[Dict[str, Any]] = []

        st = {
            "language": lang,
            "input_file": str(in_file),
            "cap": cap,
            "examples_seen": 0,
            "examples_skipped_no_cfs": 0,
            "examples_skipped_too_few_valid_cfs": 0,
            "examples_skipped_identical_cfs": 0,
            "examples_skipped_chosen_missing_after_forced_reject": 0,
            "examples_skipped_no_softflip_rejected": 0,
            "examples_skipped_pred_missing": 0,
            "pairs_written": 0,
            "forced_reject_used": 0,
            "reward_records_rewritten": 0,
        }

        for obj in read_jsonl(in_file):
            if cap is not None and n_written >= cap:
                break

            st["examples_seen"] += 1

            # Build prompt + original sentence + ids
            if args.task == "xnli":
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
                original_sentence = premise

            elif args.task == "sib200":
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
                original_sentence = text

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
                original_sentence = text

            if prepend_sys and system_instr:
                prompt = system_instr + "\n\n" + prompt

            # --- rewrite rewards in obj (optional) ---
            cfs_all = obj.get("counterfactual", {}).get(lang, [])
            if not cfs_all:
                st["examples_skipped_no_cfs"] += 1
                if args.rewrite_reward_root:
                    rewritten_reward_records.append(obj)
                    st["reward_records_rewritten"] += 1
                continue

            cfs_all = list(cfs_all)
            rewrite_n = len(cfs_all) if args.rewrite_all_cfes else min(len(cfs_all), int(args.max_cfes))
            cfs_head = cfs_all[:rewrite_n]
            cfs_tail = cfs_all[rewrite_n:]

            new_head = []
            for cf in cfs_head:
                cf2 = dict(cf)
                t_norm = (cf2.get("text") or "").strip()
                if t_norm == "":
                    new_head.append(cf2)
                    continue
                try:
                    new_flip, soft_flip, cf_pred_id = recompute_flip_rewards(
                        task=args.task,
                        cf=cf2,
                        orig_pred_id=int(orig_pred_id),
                        target_pred_id=int(target_pred_id),
                    )
                    cf2["reward_flip"] = float(new_flip)
                    cf2["reward_soft_flip"] = float(soft_flip)
                    cf2["_pred_id"] = int(cf_pred_id)
                except Exception:
                    st["examples_skipped_pred_missing"] += 1
                new_head.append(cf2)

            obj.setdefault("counterfactual", {})
            obj["counterfactual"][lang] = new_head + cfs_tail

            if args.rewrite_reward_root:
                rewritten_reward_records.append(obj)
                st["reward_records_rewritten"] += 1

            # --- build scored list for selection (max_cfes) ---
            cfs_for_select = list(obj.get("counterfactual", {}).get(lang, []))[: int(args.max_cfes)]

            scored: List[Dict[str, Any]] = []
            for cf in cfs_for_select:
                t_raw = (cf.get("text") or "")
                t_norm = t_raw.strip()
                if t_norm == "":
                    continue

                if "reward_flip" not in cf or "reward_soft_flip" not in cf:
                    try:
                        new_flip, soft_flip, cf_pred_id = recompute_flip_rewards(
                            task=args.task,
                            cf=cf,
                            orig_pred_id=int(orig_pred_id),
                            target_pred_id=int(target_pred_id),
                        )
                        cf = dict(cf)
                        cf["reward_flip"] = float(new_flip)
                        cf["reward_soft_flip"] = float(soft_flip)
                        cf["_pred_id"] = int(cf_pred_id)
                    except Exception:
                        continue

                total, used_rewards = compute_total_reward(
                    cf=cf,
                    lang=lang,
                    tgt_lang=args.tgt_lang,
                    components=args.reward_components,
                    weights=weight_map,
                )

                scored.append({
                    "text_raw": t_raw,
                    "text_norm": t_norm,
                    "total": float(total),
                    "soft_flip": float(cf.get("reward_soft_flip", 0.0)),
                    "meta": {
                        "gen_id": cf.get("gen_id", None),
                        "text": t_norm,
                        "pred": cf.get("pred", None),
                        "pred_text": cf.get("pred_text", None),
                        "pred_id": cf.get("_pred_id", None),
                        "reward_total": float(total),
                        "reward_soft_flip": float(cf.get("reward_soft_flip", 0.0)),
                        "reward_flip_recomputed": float(cf.get("reward_flip", 0.0)),
                        **used_rewards,
                        "reward_edit_raw": cf.get("reward_edit", None),
                        "reward_align_raw": cf.get("reward_align", None),
                        "reward_aug_flip_raw": cf.get("reward_aug_flip", None),
                        "reward_aug_noflip_raw": cf.get("reward_aug_noflip", None),
                    }
                })

            if len(scored) < 2:
                st["examples_skipped_too_few_valid_cfs"] += 1
                continue

            # Deduplicate by text_norm; keep max total per text
            best_by_text: Dict[str, Dict[str, Any]] = {}
            for e in scored:
                k = e["text_norm"]
                if (k not in best_by_text) or (e["total"] > best_by_text[k]["total"]):
                    best_by_text[k] = e
            uniq = list(best_by_text.values())
            uniq.sort(key=lambda x: x["total"], reverse=True)

            if len(uniq) < 2:
                st["examples_skipped_too_few_valid_cfs"] += 1
                continue

            def finalize_text(t: str) -> str:
                return ensure_edit_tags(t) if force_tags else t

            orig_norm = (original_sentence or "").strip()
            has_exact_orig = (orig_norm != "") and any(e["text_norm"] == orig_norm for e in uniq)

            # Step 1: chosen=max_total, rejected=min_total among softflip==1
            chosen_entry = uniq[0]
            softflip_candidates = [e for e in uniq if float(e.get("soft_flip", 0.0)) == 1.0]
            if not softflip_candidates:
                st["examples_skipped_no_softflip_rejected"] += 1
                continue
            rejected_entry = min(softflip_candidates, key=lambda x: x["total"])

            chosen_text_raw = chosen_entry["text_norm"]
            rejected_text_raw = rejected_entry["text_norm"]
            chosen_meta = chosen_entry["meta"]
            rejected_meta = rejected_entry["meta"]
            chosen_val = float(chosen_entry["total"])
            rejected_val = float(rejected_entry["total"])
            reject_reason = "reject=min_total_reward_among_softflip1_after_dedup"

            # Step 2: override if any exact-original text exists
            if has_exact_orig:
                rejected_text_raw = orig_norm
                rejected_meta = {"text": orig_norm, "reward_total": None, "gen_id": "original_sentence"}
                rejected_val = float("-inf")
                st["forced_reject_used"] += 1

                chosen_override = None
                for e in uniq:
                    if e["text_norm"] != orig_norm:
                        chosen_override = e
                        break
                if chosen_override is None:
                    st["examples_skipped_chosen_missing_after_forced_reject"] += 1
                    continue

                chosen_text_raw = chosen_override["text_norm"]
                chosen_meta = chosen_override["meta"]
                chosen_val = float(chosen_override["total"])
                reject_reason = "reject=explicit_original_sentence; chosen=max_total_non_original"

            # Step 3: identical skip
            if chosen_text_raw.strip() == rejected_text_raw.strip():
                st["examples_skipped_identical_cfs"] += 1
                continue

            out_obj: Dict[str, Any] = {
                "task": args.task,
                "language": lang,
                "index": obj.get("index", None),
                "prompt": prompt,
                "chosen": finalize_text(chosen_text_raw),
                "rejected": finalize_text(rejected_text_raw),
                "chosen_meta": chosen_meta,
                "rejected_meta": rejected_meta,
                "pair_note": (
                    f"dedup+rank: chosen=max_total({float(chosen_val):.6f}); "
                    f"rejected={reject_reason}({float(rejected_val):.6f}); "
                    f"max_cfes={args.max_cfes}"
                ),
            }

            if args.task == "xnli":
                out_obj["premise"] = premise
                out_obj["hypothesis"] = hypothesis
                out_obj["orig_pred"] = int(orig_pred_id)
                out_obj["target_pred"] = int(target_pred_id)
            else:
                out_obj["text"] = original_sentence
                out_obj["orig_pred"] = int(orig_pred_id)
                out_obj["target_pred"] = int(target_pred_id)

            records_out.append(out_obj)
            n_written += 1
            st["pairs_written"] += 1



        # If we did not reach cap (max_per_lang), resample existing pairs to fill.
        # This samples WITH replacement from already-constructed pairs for this language.
        if cap is not None and len(records_out) < int(cap):
            need = int(cap) - len(records_out)
            if len(records_out) == 0:
                # Nothing to sample from; keep empty.
                pass
            else:
                extra = random.choices(records_out, k=need)
                # Copy and annotate so downstream can identify resampled items.
                extra2 = []
                for r in extra:
                    rr = dict(r)
                    rr["pair_note"] = (rr.get("pair_note", "") + f"; resampled=1; cap={int(cap)}").strip("; ")
                    extra2.append(rr)
                records_out.extend(extra2)
                st["pairs_resampled"] = st.get("pairs_resampled", 0) + int(need)

                # Keep stats consistent.
                st["pairs_written"] = len(records_out)
                n_written = len(records_out)


        # write DPO output
        out_file = out_task_dir / f"{lang}.jsonl"
        write_jsonl(out_file, records_out)
        lang_stats[lang] = st

        print(f"[{args.task}] lang={lang} saved {len(records_out)} pairs -> {out_file}")
        print(
            f"[{args.task}] lang={lang} stats: seen={st['examples_seen']} "
            f"pairs_written={st['pairs_written']} "
            f"skip_no_cfs={st['examples_skipped_no_cfs']} "
            f"skip_too_few_valid_cfs={st['examples_skipped_too_few_valid_cfs']} "
            f"skip_identical_cfs={st['examples_skipped_identical_cfs']} "
            f"skip_no_softflip_rejected={st['examples_skipped_no_softflip_rejected']} "
            f"skip_pred_missing={st['examples_skipped_pred_missing']} "
            f"skip_chosen_missing_after_forced_reject={st['examples_skipped_chosen_missing_after_forced_reject']} "
            f"forced_reject_used={st['forced_reject_used']} "
            f"reward_rewritten={st['reward_records_rewritten']}"
        )

        # write rewritten reward jsonl under reward_root (optional)
        if args.rewrite_reward_root:
            out_reward_dir = Path(args.reward_root) / args.task / args.split
            out_reward_file = out_reward_dir / f"{lang}_{args.rewrite_reward_suffix}.jsonl"
            write_jsonl(out_reward_file, rewritten_reward_records)
            print(f"[{args.task}] lang={lang} rewrote rewards -> {out_reward_file}")

    stats_obj = {
        "task": args.task,
        "split": args.split,
        "out_root": str(out_root),
        "languages": args.languages,
        "language_stats": lang_stats,
    }
    with (out_root / "stats.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(stats_obj, f, allow_unicode=True)

    summary["language_stats"] = lang_stats
    with (out_root / "summary.yaml").open("w", encoding="utf-8") as f:
        yaml.dump(summary, f, allow_unicode=True)


if __name__ == "__main__":
    run(get_parser().parse_args())
