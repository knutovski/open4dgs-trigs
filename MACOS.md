# Open4DGS on macOS (Apple Silicon)

This fork runs the TRiGS trainer (`open4dgs-trigs`) and the main Open4DGS-Core preparation steps on an
Apple Silicon Mac without an NVIDIA GPU, using the PyTorch **MPS** backend. CUDA code paths are kept: on an
NVIDIA host the fork behaves like upstream.

> **Status (20 Nov 2026):** training, evaluation and PLY export run end-to-end on MPS with a small
> synthetic dataset. Preparation models (RoMa v2, MEMFOF) run on MPS. A full run on real footage
> (extraction → calibration → sync → point preparation → training) is **not yet tested**; see
> [Limitations](#known-limitations).

Tested on: Mac Studio, Apple M5 Max, 128 GB unified memory, macOS 27.2.1, Python 3.13.17, PyTorch 2.15.0,
COLMAP 4.2.1 (Homebrew, no GPU), uv 0.13.5.

Upstream base: `ASU-NEM-Labs/open4dgs-trigs@ed7a84a`, `ASU-NEM-Labs/open4dgs-core@3cac7ba`
(both 18 Nov 2026); port branch `macos-port` in both repositories.

## 1. Install

Prerequisites: Xcode Command Line Tools, [uv](https://docs.astral.sh/uv/), git. For preparation also
[Homebrew](https://brew.sh) with `brew install colmap` and `ffmpeg`/`ffprobe` on the `PATH`.

Use one workspace folder holding both forks side by side. Everything else (virtual environment,
Python, model caches) is created inside it.

```bash
mkdir -p ~/Projekte/open4dgs-mac && cd ~/Projekte/open4dgs-mac
git clone -b macos-port <your fork of open4dgs-trigs> open4dgs-trigs
git clone -b macos-port <your fork of open4dgs-core>  open4dgs-core
open4dgs-trigs/macos/install.sh            # trainer only
open4dgs-trigs/macos/install.sh --prep     # also RoMa v2 + MEMFOF for open4dgs-core/prep
```

`install.sh` clones [mpsplat](https://github.com/zacsmms/mpsplat) at a pinned commit, creates `.venv`
(Python 3.13 managed by uv inside the workspace) and installs:

| Package | macOS choice |
| --- | --- |
| `torch` / `torchvision` | 2.15.0 / 0.29.2 from PyPI (MPS build). Upstream pins 2.10.0+cu128. |
| `gsplat` | **mpsplat** (same `gsplat` package and API as gsplat 1.5.3, Metal kernels). No CUDA build. |
| `pytorch3d` | not installed; replaced by `trigs.compat.ops` |
| `fused_ssim` | not installed (CUDA only); replaced by `trigs.compat.ops.ssim_fused_compatible` |
| everything else | exact versions from upstream `uv.lock` (`macos/requirements-mac.txt`), e.g. gsply 0.2.16, pycolmap 4.0.3 |

It also runs `macos/fix_libomp.sh`: torch and pycolmap each bundle `libomp.dylib`, and importing both
aborts Python with `OMP: Error #15`. The script points pycolmap's copy at torch's (inside the venv only).
**Re-run it after reinstalling torch or pycolmap.** Do not use `KMP_DUPLICATE_LIB_OK=TRUE` instead.

Model weights (`--prep`, downloaded on first use, ~1.4 GB) go to `<workspace>/.cache` when you
`source open4dgs-trigs/macos/env.sh` first.

## 2. Check the installation

```bash
cd ~/Projekte/open4dgs-mac && source open4dgs-trigs/macos/env.sh
TRIGS_DEVICE=cpu .venv/bin/python -m pytest open4dgs-trigs/tests -q   # CPU
.venv/bin/python -m pytest open4dgs-trigs/tests -q                    # MPS (auto)
```

Result on the test machine: 240 passed, 5 failed, 3 skipped (CPU and MPS identical) before the compat and
MPS rasterization tests were added; those (17 + 2) pass. The 5 failures fail identically with the
**unmodified upstream code** in the same environment (test issues, not port issues):
`test_colmap_mask_loader` (2: expects `.png`, loader default is `.jpg`),
`test_ply_pipeline::test_ply_frames_build_current_canonical_contract` and
`::test_ply_target_count_is_applied_before_initialization` (patch `sys.modules["gsply"]` after `GSTensor`
was already imported), `::test_ply_target_helpers_are_exact_and_reproducible` (depends on a random subset;
whether it passes with upstream torch 2.10 is untested).

## 3. Quick end-to-end run (synthetic data)

```bash
.venv/bin/python open4dgs-trigs/macos/make_synth_dataset.py data/synth01      # 6 cams x 10 frames, 480x270
.venv/bin/trigs-train-colmap --data-dir data/synth01 --npz-path data/synth01/keyframes.npz \
    --output-dir runs/synth01 --start-frame 0 --end-frame 9 --keyframe-step 3 --config.n-steps 1000
.venv/bin/trigs-eval-colmap --data-dir data/synth01 --runs runs/synth01 --step -1 \
    --out-dir runs/synth01/eval --shutter captured
.venv/bin/trigs-export-plys --ckpt-path runs/synth01/trigs_ckpt.pt --output-dir runs/synth01/plys \
    --n-frames 10 --fps 30
.venv/bin/python open4dgs-trigs/macos/render_ply_compare.py data/synth01 runs/synth01/plys compare.png
```

The generator renders a known scene (static room plus a moving, rotating sphere) with mpsplat and writes
`sparse/0`, `images/camXX/`, and a `keyframes.npz` in the format of `prep-roma-trigs-memory.py`.

## 4. Real captures: preparation on macOS

Follow upstream [docs/run-pipeline.md](https://github.com/ASU-NEM-Labs/open4dgs-core/blob/main/docs/run-pipeline.md).
macOS specifics:

- `--device` defaults to `auto` (CUDA → MPS → CPU) in `prep-roma-trigs-memory.py`, `calculate-sync.py`,
  `visual-epipolar-sync.py`, `generate_masks.py`, `match_pairs_scaled.py` and the `audit-*` scripts.
- `build-colmap-ref-trigs.py` detects COLMAP 4 option names (`FeatureExtraction.*`/`FeatureMatching.*`)
  and uses CPU SIFT when COLMAP has no GPU support (Homebrew build). Feature extraction is slower than
  with CUDA.
- **Cameras without timecode** (e.g. DJI Osmo Pocket 3): `extract_synced_sequence.py` needs embedded
  timecode. Instead, align the clips frame-accurately (e.g. by audio in an editor), export one file per
  camera with identical start and length, unchanged resolution and frame rate, no stabilisation, and use
  `prep/extract/extract_synthetic_multiview.py`. `calculate-sync.py` then solves the remaining sub-frame
  offsets visually.
- Rolling-shutter readout of other cameras is unknown upstream (15 ms is the Sony RX0 II at 60 fps).
  `calculate-sync.py --rs-mode report` and `prep/sync/audit-rolling-shutter.py` estimate it.
- Default point-prep route in this repo's README (`--roma-input-dir`, `trigs.init.roma_combine`) does not
  match the current trainer (missing `accelerations`, `source_frames`); use the `prep-roma-trigs-memory.py`
  NPZ (upstream issue, not macOS-specific).

## 5. What the port changes

| Area | Change |
| --- | --- |
| `src/trigs/compat/device.py` | device selection (`auto`, `TRIGS_DEVICE`), synchronize / empty-cache / seeding / RNG-state / OOM helpers, `f64()` (float64 on the CPU for MPS tensors) |
| `src/trigs/compat/ops.py` | `knn_points` (chunked brute force; exact scipy KD-tree for large inputs), `sample_farthest_points` (batched, `lengths`, per-batch `K`), `ssim_fused_compatible` (reproduces `fused_ssim(padding="valid")`). With pytorch3d installed and CUDA tensors, pytorch3d is used. |
| `src/trigs/**` | hard-coded `"cuda"`/`torch.cuda.*` (60 places) → helpers; device defaults `"auto"`; float64 math moved off MPS (13 places); float64 relocation kernel runs on the CPU for MPS tensors |
| `open4dgs-core/prep/device_compat.py` | same device selection for prep scripts |
| `open4dgs-core/prep/align/build-colmap-ref-trigs.py` | COLMAP 3/4 option names, CPU fallback |

Environment variables: `TRIGS_DEVICE` (`cpu`, `mps`, `cuda:N`), `TRIGS_KNN_BUDGET` (default 2^28 distances
per chunk), `TRIGS_KNN_KDTREE_MIN_PAIRS` (default 2^32), `TRIGS_COMPAT_FORCE_TORCH=1` (ignore pytorch3d),
`PYTORCH_ENABLE_MPS_FALLBACK` (not needed so far; keep 0 to see unsupported ops).

## 6. Measurements (M5 Max, 128 GB; measured 19–20 Nov 2026)

| What | Result |
| --- | --- |
| Rasterization MPS vs CPU (500 splats, SH 3) | image max diff ≤ 4.1e-3, rel. gradient diff ≤ 1.3e-3 |
| `knn_points`, self query, K=11 | 0.1 M pts 0.03 s, 1 M 0.36 s, 4 M 1.36 s (KD-tree; brute force on MPS: 1 M in 444 s) |
| `sample_farthest_points`, 1 M points | → 1,000: 0.28 s, → 10,000: 2.66 s on MPS (CPU 2.05 s / 20.4 s) |
| Training, synthetic (6 cams, 10 frames, 480×270, 20k primitives) | 1,000 steps in 86.6 s total (~12.5 steps/s, 6 cameras per step), peak memory footprint 8.2 GB |
| Quality after 1,000 steps | `trigs-eval-colmap` 34.8 dB (42 pairs; 6.0 dB at step 0); exported PLYs re-rendered vs ground truth 30.5 dB mean |
| RoMa v2 (precise), 480×270 pair | 1.08 s (first call 5.5 s); epipolar error vs known cameras median 0.11 px |
| MEMFOF, 480×270 triplet | 0.08 s (first call 1.3 s) |

Real-footage numbers (resolution, camera count, prep and training time, memory): **not yet measured.**

## Known limitations

- Untested so far: full preparation on real footage, long sequences / chunked training, masks (SAM 2,
  BGMv2), `trigs-train-plys` and `trigs-train-selfcap` paths, the interactive viewer on MPS.
- Not ported: EasyVolCap / 4K4D camera refinement (needs Torch 2.1 + CUDA 11.8 + tiny-cuda-nn),
  VGGT alignment. Neither is part of the main pipeline.
- No distributed (multi-GPU) training on macOS.
- COLMAP from Homebrew has no GPU support; SIFT runs on the CPU.
- RoMa v2 requests bfloat16 autocast for CUDA only; on MPS it runs in float32 (correct, possibly slower).
- mpsplat routes rolling-shutter/ftheta rasterization through PyTorch reference code; TRiGS uses its own
  rolling-shutter model and does not depend on that path.
- Two `device_synchronize()` calls per training step now also run on MPS (they were CUDA-only); their cost
  has not been measured separately.
