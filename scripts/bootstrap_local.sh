#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv_311"

# Point Git at the tracked hooks directory so every clone gets the public-push
# gate without a separate install step, and so edits to `.githooks/pre-push`
# take effect immediately rather than after a reinstall (audit C-30).
git -C "${ROOT_DIR}" config core.hooksPath .githooks

python3.11 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip
python -m pip install -r "${ROOT_DIR}/backend/requirements.txt"
npm --prefix "${ROOT_DIR}/frontend" ci

printf '\nBootstrap complete.\n'
printf 'Activate with: source %s/bin/activate\n' "${VENV_DIR}"
printf 'Verify with: make verify-local\n'
