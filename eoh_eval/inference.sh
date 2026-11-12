#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OUTPUT_PATH="${OUTPUT_PATH:-${SCRIPT_DIR}/../outputs/eoh}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
exec bash "${SCRIPT_DIR}/../eval/inference.sh" "$@" --data_name '["eoh_eval.jsonl"]'
