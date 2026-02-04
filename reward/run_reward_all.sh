#!/usr/bin/env bash
set -euo pipefail

langs=(en ar de ru sw vi zh)

IN_ROOT="../data_qwen_sample"
OUT_ROOT="../data_qwen_reward"
LOCAL_MODEL_DIR="/root/autodl-tmp/model/Qwen3-8B"
NLLB_MODEL_NAME="/root/autodl-tmp/model/nllb-200-distilled-1.3B"


for lang in "${langs[@]}"; do
  echo "=============================="
  echo "[run] lang=${lang}  start=$(date '+%Y-%m-%d %H:%M:%S')"
  echo "=============================="

  python process_all_rewards.py \
    --in_root "${IN_ROOT}" \
    --out_root "${OUT_ROOT}" \
    --local_model_dir "${LOCAL_MODEL_DIR}" \
    --nllb_model_name "${NLLB_MODEL_NAME}" \
    --langs "${lang}"
    #--max_examples 10
  echo "[done] lang=${lang} end=$(date '+%Y-%m-%d %H:%M:%S')"
done
