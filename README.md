# MACRO

Multilingual counterfactual generation with preference optimization.


![Overview](figs/macro.png)

## Overview


MACRO is a framework for multilingual counterfactual generation in text classification. Given an input and a model prediction, it aims to produce a minimally edited counterfactual that changes the prediction while preserving as much of the original meaning and form as possible.

The pipeline consists of three stages: counterfactual candidate sampling, preference pair ranking, and DPO-based preference alignment. It first generates multiple counterfactual candidates, then ranks them with scores, and finally trains the model to prefer better counterfactuals over worse ones.



## Setup

Install dependencies:


```bash
pip install transformers==4.56.2
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install soxr accelerate datasets pyyaml wandb peft trl
pip install -r requirements.txt
```

## Pipeline

### 1. Preprocess
Generate model predictions, then build target data for later counterfactual construction.

```bash
python preprocess/generate_predictions.py \
  --task <sib200|taxi1500> \
  --split <train|validation|test> \
  --lang <en|ar|de|ru|sw|vi|zh> \
  --out_dir <output_dir> \
  --local_model_dir <model_dir> \
  [--resume]

python preprocess/make_target.py \
  --data_root <prediction_dir> \
  --repair_model_dir <model_dir> \
  --langs <en|ar|de|ru|sw|vi|zh> \
  --datasets <sib200|taxi1500> \
  --splits <train|validation|test>
```

### 2. Sample
Generate counterfactual candidates for a dataset, split, and language.

```bash
python sample/generate_counterfactual.py \
  --language <en|ar|de|ru|sw|vi|zh> \
  --dataset_name <sib200|taxi1500> \
  --split <train|validation|test> \
  --num_generations <int>
```

### 3. Rank and train
First rank sampled candidates into preference pairs, then train with DPO.

```bash
python train/ranking.py \
  --task <sib200|taxi1500> \
  --split <train|validation|test> \
  --reward_root <reward_dir> \
  --out_root <triple_dir> \
  --languages <en|ar|de|ru|sw|vi|zh> \
  --reward_components <flip|edit|aug> \
  --reward_weights <name=value ...> \
  --tag <run_tag>

python train/dpo.py \
  --task <sib200|taxi1500> \
  --triple_root <triple_dir> \
  --languages <en|ar|de|ru|sw|vi|zh> \
  --model_name_or_path <model_dir_or_hf_name> \
  --output_dir <run_dir> \
  --run_name <run_name> \
  --adapter_root <adapter_root> \
  --adapter_name <adapter_name> 
```

### 4. Evaluate
Generate outputs from a trained adapter, then run automatic evaluation.

```bash
python eval/generate_counterfactual_macro.py \
  --dataset_name <sib200|taxi1500> \
  --language <en|ar|de|ru|sw|vi|zh> \
  --adapter_path <adapter_path> \
  --out_root <output_dir> \
  --pred_root <prediction_dir>

python eval/eval_all.py \
  --data_root <generation_dir> \
  --task <sib200|taxi1500> \
  --split <test|validation|train> \
  --langs <all|lang1 lang2 ...> \
  --do_ppl --do_sim --do_edit \
  --classifier_model_dir <model_dir_or_hf_name> \
  --output_csv <results_csv>
```

## Counterfactual data augmentation

Does adding generated counterfactuals to the fine-tuning data improve
classification accuracy over fine-tuning on the original data alone? A LoRA
classifier is trained twice per (model, dataset) — once on the original train
split (`baseline`), once on the original split plus successful counterfactuals
for the same inputs (`augmented`) — and both are evaluated on the test set with
the prompt and constrained greedy decoding imported from
`preprocess/generate_predictions.py`. Run from the repository root.

Both conditions are trained from a `{prompt, completion}` jsonl, where `prompt`
is built with `build_prompt_sib200` / `build_prompt_taxi1500` from
`preprocess/generate_predictions.py` and `completion` is the label word. For the
`augmented` condition, add one successful counterfactual per original example
(`reward_flip == 1.0`), paired with its target label.

```bash
# 1. Fine-tune one LoRA classifier per condition
python counterfactual_augmentation/scripts/train_classifier_lora.py \
  --train_file <train_jsonl> \
  --base_model <model_dir_or_hf_name> \
  --output_dir <adapter_dir> \
  [--validation_file <validation_jsonl>] \
  [--num_train_epochs 2] [--lora_r 128] [--auto_find_batch_size]

# 2. Evaluate each fine-tuned adapter on the test set
python counterfactual_augmentation/scripts/eval_after_finetune.py \
  --base_model <model_dir_or_hf_name> \
  --adapter_path <adapter_dir> \
  --model_tag <qwen4|qwen8|gemma4|gemma12> \
  --condition <baseline_sft|augmented_sft> \
  --dataset <sib200|taxi1500> \
  --pred_root <prediction_dir> \
  --out_csv counterfactual_augmentation/results/<model>_<dataset>_<condition>_eval.csv
```

Step 2 writes one CSV per run:
`model, condition, dataset, lang, n, n_correct, acc_pct`. The pooled comparisons
in `results/combined_cda_before_after.csv` and
`results/summary_cda_by_model_dataset.csv` aggregate those per-run CSVs against
the zero-shot accuracy of the untuned base model, and report a two-proportion
`z_aug_vs_baseline` with `significant_at_p05`.
