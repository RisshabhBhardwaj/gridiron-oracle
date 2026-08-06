#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv_311"

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  echo ".venv_311 is missing. Run 'make bootstrap-local' first." >&2
  exit 1
fi

source "${VENV_DIR}/bin/activate"

cd "${ROOT_DIR}"
python -m py_compile \
  backend/app/main.py \
  backend/app/core/tracing.py \
  backend/app/services/projection.py \
  backend/app/services/runtime_status.py \
  backend/app/services/backtest.py \
  backend/app/api/backtest.py \
  backend/app/api/explain.py \
  backend/app/api/predict.py \
  backend/app/core/config.py \
  ml/shap_service.py

pytest backend/tests/test_api.py -q
npm --prefix frontend run build
npm --prefix frontend run test
