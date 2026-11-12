#!/usr/bin/env bash
# FrontierOR one-shot evaluation of one self-hosted model behind an OpenAI-compatible endpoint.
#
#   FRONTIEROR_DIR=/path/to/FrontierOR MODEL=sdrl-qwen3-32b \
#   BASE_URL=http://localhost:8000/v1 PROMPT_MODE=sdrl bash frontieror_eval/run_frontieror.sh
#
# Defaults follow the FrontierOR paper protocol: all 180 tasks, tiny gate + large_1..5,
# 3600 s per run, up to 5 debug retries, 1 CPU core per candidate program. Resumable:
# rerun with the same OUT and finished (paper, model) pairs are skipped.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${FRONTIEROR_DIR:?set FRONTIEROR_DIR to a FrontierOR checkout with frontieror_selfhosted.patch applied}"
: "${MODEL:?set MODEL to the vLLM --served-model-name}"
: "${BASE_URL:?set BASE_URL, e.g. http://localhost:8000/v1}"
: "${GRB_LICENSE_FILE:?set GRB_LICENSE_FILE to a Gurobi license}"
export GRB_LICENSE_FILE

PROMPT_MODE="${PROMPT_MODE:-sdrl}"
INSTANCES="${INSTANCES:-tiny large_1 large_2 large_3 large_4 large_5}"
TIME_LIMIT="${TIME_LIMIT:-3600}"
DEBUG_RETRIES="${DEBUG_RETRIES:-5}"
# RESERVE>0 is an analysis variant, not the paper protocol: the program is told
# --time_limit (TIME_LIMIT - RESERVE) while the hard kill stays at TIME_LIMIT.
RESERVE="${RESERVE:-0}"
CPUS="${CPUS:-$(nproc)}"
PAPER_WORKERS="${PAPER_WORKERS:-12}"
INSTANCE_WORKERS="${INSTANCE_WORKERS:-$(( CPUS > 8 ? CPUS - 4 : CPUS ))}"
OUT="${OUT:-${FRONTIEROR_DIR}/eval/${MODEL}_${PROMPT_MODE}_reserve${RESERVE}}"
TMP_DIR="${TMP_DIR:-${OUT}/tmp}"

export FRONTIER_OR_ONESHOT_CONFIG="${ONESHOT_CONFIG:-${SCRIPT_DIR}/oneshot_selfhosted.yaml}"
export FRONTIER_OR_LLM_BASE_URL="$BASE_URL"
export FRONTIER_OR_LLM_API_KEY="${API_KEY:-EMPTY}"
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-unused-self-hosted}"
export FRONTIER_OR_LLM_TIMEOUT="${LLM_TIMEOUT:-1800}"
export FRONTIER_OR_CANDIDATE_TIME_RESERVE="$RESERVE"
export PYTHONUNBUFFERED=1

case "$PROMPT_MODE" in
  # SDRL strategy-routing system prompt (+ runtime requirements), FrontierOR user prompt,
  # and "/no_think" for Qwen3 chat templates.
  sdrl)   export FRONTIER_OR_SYSTEM_PROMPT_FILE="${SCRIPT_DIR}/sdrl_frontieror.system.txt" FRONTIER_OR_USER_SUFFIX=/no_think ;;
  # Unmodified FrontierOR prompt (no system prompt, no suffix).
  native) unset FRONTIER_OR_SYSTEM_PROMPT_FILE FRONTIER_OR_USER_SUFFIX ;;
  *) echo "unknown PROMPT_MODE=$PROMPT_MODE (sdrl|native)" >&2; exit 2 ;;
esac

mkdir -p "$OUT" "$TMP_DIR"
curl -sf -m 20 "$BASE_URL/models" > /dev/null || { echo "endpoint not reachable: $BASE_URL" >&2; exit 2; }
echo "model=$MODEL prompt=$PROMPT_MODE reserve=$RESERVE time_limit=$TIME_LIMIT out=$OUT"

cd "$FRONTIEROR_DIR"
# shellcheck disable=SC2086
"${PYTHON:-python3}" -u one_shot_eval.py --models "$MODEL" --instances $INSTANCES \
  --max_debug_retries "$DEBUG_RETRIES" --time_limit "$TIME_LIMIT" \
  --paper_workers "$PAPER_WORKERS" --model_workers 1 --instance_workers "$INSTANCE_WORKERS" \
  --exec-mode bare --cpus 1 --temp-dir "$TMP_DIR" \
  --output-root "$OUT/eval_tasks" --results_csv "$OUT/eval_results_${MODEL}.csv" \
  --api-cost-csv "$OUT/api_cost_${MODEL}.csv" ${SKIP_EXISTING---skip-existing} ${EXTRA_ARGS:-} \
  2>&1 | tee -a "$OUT/eval.log"
