#!/usr/bin/env bash
# torch and pycolmap each ship libomp.dylib; loading both aborts with "OMP: Error #15".
# Point pycolmap's copy at torch's (relative symlink inside the venv). Idempotent; re-run after
# reinstalling torch or pycolmap.
set -euo pipefail
WS="$(cd "$(dirname "$0")/../.." && pwd)"
SP=$("$WS/.venv/bin/python" -c "import site; print(site.getsitepackages()[0])")
T="$SP/pycolmap/.dylibs/libomp.dylib"
[ -e "$SP/torch/lib/libomp.dylib" ] || { echo "torch/lib/libomp.dylib missing"; exit 1; }
[ -e "$T" ] || [ -L "$T" ] || { echo "pycolmap has no bundled libomp - nothing to do"; exit 0; }
if [ -L "$T" ]; then echo "already linked: $(readlink "$T")"; else
  mv "$T" "$T.pycolmap-orig"; ln -s ../../torch/lib/libomp.dylib "$T"; echo "linked $T -> torch libomp"
fi
