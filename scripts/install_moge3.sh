#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON:-python}"
WORK_DIR="$(mktemp -d /tmp/vistadream-moge3.XXXXXX)"

cleanup() {
    rm -rf -- "${WORK_DIR}"
}
trap cleanup EXIT

"${PYTHON_BIN}" -c "import torch, triton"

git clone --filter=blob:none https://github.com/JeffreyXiang/FlexGEMM.git \
    "${WORK_DIR}/FlexGEMM"
git -C "${WORK_DIR}/FlexGEMM" checkout \
    b2fadb29d41846c7981ade6801ffc689fae119cf
git -C "${WORK_DIR}/FlexGEMM" apply \
    "${PROJECT_DIR}/patches/flexgemm-triton33.patch"

"${PYTHON_BIN}" -m pip install --force-reinstall --no-deps --no-build-isolation \
    "${WORK_DIR}/FlexGEMM"
"${PYTHON_BIN}" -m pip install --no-deps \
    "git+https://github.com/EasternJournalist/utils3d-moge.git@62f09d58509485564e24d5d9f6aac9ee9ebc0c37" \
    "git+https://github.com/microsoft/MoGe.git@74fbce054ebed49800de42d0ad0e83495065719a"

cd "${PROJECT_DIR}"
"${PYTHON_BIN}" -c \
    "from ops.legos.inpaints.moge3 import MoGe3_Tool; print('MoGe-3 imports: OK')"
