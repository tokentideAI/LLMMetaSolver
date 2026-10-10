#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

PYTHON="${PYTHON:-python3}"
MODEL_NAME="${MODEL_NAME:-}"
MODEL_LIST="${MODEL_LIST:-}"
TOKENIZER_NAME="${TOKENIZER_NAME:-}"
DATA_PATH="${DATA_PATH:-${SCRIPT_DIR}/../test_data}"
OUTPUT_PATH="${OUTPUT_PATH:-${SCRIPT_DIR}/inference_outputs}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MODEL_SOURCE="${MODEL_SOURCE:-auto}"
PROMPT_NAME="${PROMPT_NAME:-sdrl_prompt}"
DECODING_TYPE="${DECODING_TYPE:-top_p}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
SEED="${SEED:-}"
TEMPERATURE="${TEMPERATURE:-}"
ENABLE_THINKING="${ENABLE_THINKING:-false}"
PARALLEL_MODE="${PARALLEL_MODE:-0}"
DRY_RUN="${DRY_RUN:-0}"

COMMON_ARGS=(
  --data_path "${DATA_PATH}"
  --output_path "${OUTPUT_PATH}"
  --model_source "${MODEL_SOURCE}"
  --prompt_name "${PROMPT_NAME}"
  --decoding_type "${DECODING_TYPE}"
  --batch_size "${BATCH_SIZE}"
  --max_tokens "${MAX_TOKENS}"
  --num_samples "${NUM_SAMPLES}"
  --enable_thinking "${ENABLE_THINKING}"
)
if [[ -n "${SEED}" ]]; then
  COMMON_ARGS+=(--seed "${SEED}")
fi
if [[ -n "${TEMPERATURE}" ]]; then
  COMMON_ARGS+=(--temperature "${TEMPERATURE}")
fi
if [[ -n "${MAX_MODEL_LEN}" ]]; then
  COMMON_ARGS+=(--max_model_len "${MAX_MODEL_LEN}")
fi
if [[ -n "${TOKENIZER_NAME}" ]]; then
  COMMON_ARGS+=(--tokenizer_name "${TOKENIZER_NAME}")
fi

if [[ -n "${MODEL_LIST}" && "${PARALLEL_MODE}" == "1" ]]; then
  [[ -f "${MODEL_LIST}" ]] || {
    echo "ERROR: model list not found: ${MODEL_LIST}" >&2
    exit 2
  }

  IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
  [[ ${#GPU_IDS[@]} -gt 0 ]] || {
    echo "ERROR: CUDA_VISIBLE_DEVICES is empty" >&2
    exit 2
  }

  mkdir -p "${OUTPUT_PATH}/launcher_logs"
  PIDS=()
  LABELS=()
  FAILURES=0
  MODEL_INDEX=0

  wait_batch() {
    local index
    for index in "${!PIDS[@]}"; do
      if wait "${PIDS[$index]}"; then
        echo "[launcher] completed: ${LABELS[$index]}"
      else
        echo "[launcher] FAILED: ${LABELS[$index]}" >&2
        FAILURES=$((FAILURES + 1))
      fi
    done
    PIDS=()
    LABELS=()
  }

  while IFS=$'\t' read -r MODEL_LABEL MODEL_PATH; do
    [[ -n "${MODEL_LABEL// }" ]] || continue
    [[ "${MODEL_LABEL}" == \#* ]] && continue
    if [[ -z "${MODEL_PATH:-}" ]]; then
      MODEL_PATH="${MODEL_LABEL}"
      MODEL_LABEL="$(basename "${MODEL_PATH}")"
    fi
    if [[ "${MODEL_PATH}" == /* && ! -d "${MODEL_PATH}" ]]; then
      echo "[launcher] skip missing local model (merge it first): ${MODEL_LABEL} -> ${MODEL_PATH}"
      continue
    fi
    GPU_ID="${GPU_IDS[$((MODEL_INDEX % ${#GPU_IDS[@]}))]}"
    LOG_FILE="${OUTPUT_PATH}/launcher_logs/${MODEL_LABEL}.log"
    echo "[launcher] ${MODEL_LABEL} -> physical GPU ${GPU_ID}"

    COMMAND=(
      "${PYTHON}" "${SCRIPT_DIR}/inference.py"
      --model_name "${MODEL_PATH}"
      --run_name "${MODEL_LABEL}"
      --tensor_parallel_size 1
      "${COMMON_ARGS[@]}"
      "$@"
    )
    if [[ "${DRY_RUN}" == "1" ]]; then
      printf '[launcher][dry-run] CUDA_VISIBLE_DEVICES=%q ' "${GPU_ID}"
      printf '%q ' "${COMMAND[@]}"
      printf '\n'
    else
      (
        set -o pipefail
        CUDA_VISIBLE_DEVICES="${GPU_ID}" "${COMMAND[@]}" 2>&1 \
          | sed -u "s/^/[${MODEL_LABEL}] /" \
          | tee "${LOG_FILE}"
      ) &
      PIDS+=("$!")
      LABELS+=("${MODEL_LABEL}")
    fi
    MODEL_INDEX=$((MODEL_INDEX + 1))

    if [[ "${DRY_RUN}" != "1" && ${#PIDS[@]} -ge ${#GPU_IDS[@]} ]]; then
      wait_batch
    fi
  done < "${MODEL_LIST}"

  if [[ "${DRY_RUN}" != "1" && ${#PIDS[@]} -gt 0 ]]; then
    wait_batch
  fi
  if [[ "${FAILURES}" -gt 0 ]]; then
    echo "ERROR: ${FAILURES} model inference jobs failed" >&2
    exit 1
  fi
  exit 0
fi

if [[ -z "${MODEL_NAME}" ]]; then
  echo "ERROR: set MODEL_NAME, or set MODEL_LIST with PARALLEL_MODE=1" >&2
  exit 2
fi

COMMAND=(
  "${PYTHON}" "${SCRIPT_DIR}/inference.py"
  --model_name "${MODEL_NAME}"
  --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}"
  "${COMMON_ARGS[@]}"
  "$@"
)
if [[ "${DRY_RUN}" == "1" ]]; then
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit 0
fi
exec "${COMMAND[@]}"
