#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bash "${PROJECT_DIR}/scripts/install_moge3.sh"
python -m pip install --no-deps \
  'git+https://github.com/Robbyant/lingbot-depth.git@f3a237e434ae987bc38281476d6cfb5df3e4d739' \
  'gradio-model3dgscamera==0.0.3'
python "${PROJECT_DIR}/scripts/patch_gradio_gs_camera.py"
