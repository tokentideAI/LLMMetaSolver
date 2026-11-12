#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ORSTAR_GUROBI_THREADS="${ORSTAR_GUROBI_THREADS:-1}"
export ORSTAR_EXEC_MAX_MEM_MB="${ORSTAR_EXEC_MAX_MEM_MB:-1000}"
exec "${PYTHON:-python3}" "${SCRIPT_DIR}/judge.py" \
  --input_path "${INPUT_PATH:-${SCRIPT_DIR}/../outputs/eoh}" \
  --exec_timeout "${EXEC_TIMEOUT:-3600}" \
  --workers "${EOH_WORKERS:-2}" "$@"
