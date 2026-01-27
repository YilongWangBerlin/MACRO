#!/usr/bin/env bash
set -euo pipefail

# ========== Config ==========
#LANGS=("en" "ar" "de" "ru" "sw" "vi" "zh")
LANGS=("en" "de" "zh")

TASKS=("xnli" "sib200")
SPLITS=("train" "test" "validation")

PRED_ROOT="../data_qwen_pred"
OUT_DIR="../data_qwen_sample"
#MODEL_DIR="/root/autodl-tmp/model/gemma-2-9b-it"
MODEL_DIR="/root/autodl-tmp/model/Qwen3-8B"


# 你要跑的 python 脚本（按你的项目实际文件名改）
PY_SCRIPT="generate_counterfactual.py"

#declare -A DEFAULT_LIMITS=(
#  ["xnli,train"]=3000
#  ["xnli,test"]=400
#  ["xnli,validation"]=500
#
#
#  ["sib200,train"]=3000
#  ["sib200,test"]=99
#  ["sib200,validation"]=99
#)
declare -A DEFAULT_LIMITS=(
  ["xnli,train"]=20
  ["xnli,test"]=5
  ["xnli,validation"]=5

  ["sib200,train"]=20
  ["sib200,test"]=5
  ["sib200,validation"]=5
)

mkdir -p "$OUT_DIR"
LOG_FILE="${OUT_DIR}/run_$(date +%y%m%d_%H%M%S).log"

echo "LOG_FILE=$LOG_FILE" | tee -a "$LOG_FILE"
echo "PY_SCRIPT=$PY_SCRIPT" | tee -a "$LOG_FILE"
echo "MODEL_DIR=$MODEL_DIR" | tee -a "$LOG_FILE"
echo "PRED_ROOT=$PRED_ROOT" | tee -a "$LOG_FILE"
echo "OUT_DIR=$OUT_DIR" | tee -a "$LOG_FILE"
echo "Start at: $(date)" | tee -a "$LOG_FILE"
echo "----------------------------------------" | tee -a "$LOG_FILE"

total_start=$(date +%s)

for task in "${TASKS[@]}"; do
  for split in "${SPLITS[@]}"; do
    for lang in "${LANGS[@]}"; do

      key="${task},${split}"
      limit="${DEFAULT_LIMITS[$key]:-0}"

      echo "[RUN] task=$task split=$split lang=$lang max_samples=$limit" | tee -a "$LOG_FILE"
      start_ts=$(date +%s)


      python "$PY_SCRIPT" \
        --dataset_name "$task" \
        --split "$split" \
        --language "$lang" \
        --pred_root "$PRED_ROOT" \
        --out_root "$OUT_DIR" \
        --model_name_or_path "$MODEL_DIR" \
        --max_samples "$limit" 2>&1 | tee -a "$LOG_FILE"

      end_ts=$(date +%s)
      elapsed=$(( end_ts - start_ts ))

      echo "[DONE] task=$task split=$split lang=$lang elapsed=${elapsed}s" | tee -a "$LOG_FILE"
      echo "----------------------------------------" | tee -a "$LOG_FILE"

    done
  done
done

total_end=$(date +%s)
total_elapsed=$(( total_end - total_start ))
echo "All done at: $(date), total_elapsed=${total_elapsed}s" | tee -a "$LOG_FILE"
