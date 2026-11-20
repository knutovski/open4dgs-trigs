#!/usr/bin/env bash
# macOS (Apple Silicon) install for open4dgs-trigs (+ optional open4dgs-core prep models).
# Expects the workspace layout described in MACOS.md:
#   <ws>/open4dgs-trigs  <ws>/open4dgs-core  <ws>/mpsplat
# Usage: open4dgs-trigs/macos/install.sh [--prep]
set -euo pipefail
WS="$(cd "$(dirname "$0")/../.." && pwd)"; cd "$WS"
source open4dgs-trigs/macos/env.sh
export UV_PYTHON_INSTALL_DIR="$WS/.uv-python"
PY=.venv/bin/python
MPSPLAT_COMMIT=5a6bf1b05f67b055135734db10134ae1caabdab1
[ -d mpsplat ] || git clone https://github.com/zacsmms/mpsplat
git -C mpsplat checkout -q "$MPSPLAT_COMMIT"
[ -x "$PY" ] || uv venv --python 3.13 .venv
uv pip install --python $PY "torch==2.15.0" "torchvision==0.29.2"
uv pip install --python $PY -e ./mpsplat                  # provides the "gsplat" package (Metal kernels)
uv pip install --python $PY --no-deps -e ./open4dgs-trigs
uv pip install --python $PY -r open4dgs-trigs/macos/requirements-mac.txt "pytest>=8.0"
if [ "${1:-}" = "--prep" ]; then
  uv pip install --python $PY "git+https://github.com/Parskatt/RoMaV2.git@v2.0.1"
  uv pip install --python $PY "git+https://github.com/msu-video-group/memfof.git@a51de9fc59c6fe20ba08e079372c7b583d58a712"
fi
open4dgs-trigs/macos/fix_libomp.sh
$PY -c "import torch, pycolmap, gsplat, trigs; print('torch', torch.__version__, '| mps', torch.backends.mps.is_available(), '| gsplat (mpsplat)', gsplat.__version__, '| trigs', trigs.__version__)"
