#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export ORSTAR_GUROBI_THREADS="${ORSTAR_GUROBI_THREADS:-1}"
export ORSTAR_CHECK_WORKERS="${ORSTAR_CHECK_WORKERS:-16}"

PYTHON="${PYTHON:-python3}"
INPUT_PATH="${INPUT_PATH:-${SCRIPT_DIR}/inference_outputs}"
EXEC_TIMEOUT="${EXEC_TIMEOUT:-300}"

exec "${PYTHON}" "${SCRIPT_DIR}/judge.py" \
  --input_path "${INPUT_PATH}" \
  --check_workers "${ORSTAR_CHECK_WORKERS}" \
  --gurobi_threads "${ORSTAR_GUROBI_THREADS}" \
  --exec_timeout "${EXEC_TIMEOUT}" \
  "$@"
