#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv_311"

python3.11 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "${ROOT_DIR}/backend/requirements.txt"
npm --prefix "${ROOT_DIR}/frontend" ci

printf '\nBootstrap complete.\n'
printf 'Activate with: source %s/bin/activate\n' "${VENV_DIR}"
printf 'Verify with: make verify-local\n'
