#!/bin/bash
set -u

mkdir -p logs

LANGS=("en" "ar" "de" "ru" "sw" "vi" "zh")
#LANGS=("en" "sw" "zh")

TASKS=("sib200" "xnli" )
SPLITS=("train" "test" "validation")


OUT_DIR="../data_qwen_pred_new"
MODEL_DIR="/root/autodl-tmp/model/Qwen3-8B"

TOTAL=$(( ${#LANGS[@]} * ${#TASKS[@]} * ${#SPLITS[@]} ))
CUR=0
START_TS=$(date +%s)
FAIL=0

format_time () {
  local s=$1
  local h=$((s/3600))
  local m=$(((s%3600)/60))
  local sec=$((s%60))
  printf "%02d:%02d:%02d" "$h" "$m" "$sec"
}

progress_bar () {
  local cur=$1
  local total=$2
  local msg="$3"
  local width=30

  local percent=$(( 100 * cur / total ))
  local filled=$(( width * cur / total ))
  local empty=$(( width - filled ))

  local bar_filled
  local bar_empty
  bar_filled=$(printf "%*s" "$filled" "" | tr ' ' '#')
  bar_empty=$(printf "%*s" "$empty" "")

  local now elapsed eta
  now=$(date +%s)
  elapsed=$(( now - START_TS ))
  if (( cur > 0 )); then
    eta=$(( elapsed * (total - cur) / cur ))
  else
    eta=0
  fi

  printf "\r[%3d%%] (%d/%d) [%-${width}s] ETA %s | %s" \
    "$percent" "$cur" "$total" "${bar_filled}${bar_empty}" "$(format_time "$eta")" "$msg" >&2
}

for task in "${TASKS[@]}"; do
  for split in "${SPLITS[@]}"; do
    for lang in "${LANGS[@]}"; do
      CUR=$((CUR + 1))
      log="logs/${task}_${split}_${lang}.log"

      progress_bar "$CUR" "$TOTAL" "Running: ${task} ${split} ${lang} -> ${log}"

      python3 generate_predictions.py \
        --task "$task" \
        --split "$split" \
        --lang "$lang" \
        --out_dir "$OUT_DIR" \
        --local_model_dir "$MODEL_DIR" \
        --resume \
        > "$log" 2>&1

      rc=$?
      if (( rc != 0 )); then
        FAIL=1
        printf "\n[WARN] Failed: %s %s %s (exit=%d). See %s\n" "$task" "$split" "$lang" "$rc" "$log" >&2
      fi
    done
  done
done

printf "\nAll tasks finished. Logs are in ./logs\n" >&2

if (( FAIL == 0 )); then
  mt_log="logs/make_target.log"
  printf "All prediction jobs succeeded. Running make_target.py -> %s\n" "$mt_log" >&2

  python3 make_target.py \
    --data_root "$OUT_DIR" \
    --repair_model_dir "$MODEL_DIR" \
    --langs "${LANGS[@]}" \
    --datasets "${TASKS[@]}" \
    --splits "${SPLITS[@]}" \
    > "$mt_log" 2>&1

  rc=$?
  if (( rc != 0 )); then
    printf "\n[WARN] make_target failed (exit=%d). See %s\n" "$rc" "$mt_log" >&2
    exit $rc
  fi
else
  printf "\n[WARN] Some prediction jobs failed, skip make_target.\n" >&2
  exit 1
fi
