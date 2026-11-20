# source macos/env.sh -- model caches inside the workspace instead of ~/.cache
_WS="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/../.." && pwd)"  # bash or zsh
export HF_HOME="$_WS/.cache/huggingface"
export TORCH_HOME="$_WS/.cache/torch"
export XDG_CACHE_HOME="$_WS/.cache"
export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-0}"
