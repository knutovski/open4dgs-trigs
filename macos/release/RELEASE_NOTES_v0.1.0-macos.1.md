# v0.1.0-macos.1 — first macOS (Apple Silicon) release

Unofficial port of [Open4DGS-Trigs](https://github.com/ASU-NEM-Labs/open4dgs-trigs) to Apple Silicon Macs (PyTorch MPS), based on upstream commit `ed7a84a`. Companion preparation fork: [knutovski/open4dgs-core](https://github.com/knutovski/open4dgs-core) (`macos-port`, based on `3cac7ba`). Not affiliated with or endorsed by the original authors. CUDA code paths are kept.

## What works on macOS
- Training (`trigs-train-colmap`), evaluation, PLY export and the interactive viewer on MPS.
- Preparation from open4dgs-core: frame extraction, COLMAP calibration (Homebrew COLMAP, CPU SIFT), sub-frame sync, point preparation with RoMa v2 and MEMFOF on MPS.
- gsplat is provided by [mpsplat](https://github.com/zacsmms/mpsplat) (Metal kernels); pytorch3d and fused_ssim are replaced by pure-PyTorch code in `trigs.compat`.

## Assets
| File | What it is |
| --- | --- |
| `trigs-0.1.0+macos.1-py3-none-any.whl` | the trainer package of this release |
| `mpsplat-1.5.3-py3-none-any.whl` | mpsplat built from commit `5a6bf1b` (provides `gsplat`), Apache-2.0 |
| `LICENSE-mpsplat.txt` | mpsplat's license |
| `requirements-macos-lock.txt` | exact versions this release was tested with (torch 2.15.0, Python 3.13) |
| `SHA256SUMS` | checksums of the files above |

## Install
See [MACOS.md](https://github.com/knutovski/open4dgs-trigs/blob/macos-port/MACOS.md#install-from-a-github-release-prebuilt-wheels): clone both forks, download the assets, run `open4dgs-trigs/macos/install.sh --from-release=<dir>`.

## Tested
Mac Studio, Apple M5 Max, 128 GB, macOS 27.2.1. Test suite: 246 passed, 3 skipped; 5 tests fail identically with the unmodified upstream code (test issues, listed in MACOS.md). End-to-end runs with a synthetic dataset and with a short real 4-camera capture (DJI Osmo Pocket 3, 1080p50).

## Known limitations
- Reconstruction quality depends on calibration and camera count; with 4 cameras the real test capture was poor (not caused by the port). See MACOS.md, section 7.
- Untested: masks (SAM 2, BGMv2), chunked training on long sequences, `trigs-train-plys` / `trigs-train-selfcap`.
- Not ported: EasyVolCap, VGGT. No multi-GPU training.
- Model weights (RoMa v2/DINOv3, MEMFOF) are not included; they download on first use under their own licenses.

## License
Trainer code: AGPL-3.0 (source in this repository). mpsplat: Apache-2.0. See THIRD_PARTY_NOTICES.md.
