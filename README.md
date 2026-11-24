> [!NOTE]
> **Unofficial macOS (Apple Silicon) port.** This is a modified fork of
> [ASU-NEM-Labs/open4dgs-trigs](https://github.com/ASU-NEM-Labs/open4dgs-trigs), changed from November 2026
> (branch `macos-port`, based on upstream commit `ed7a84a`) to run on Apple Silicon Macs with the PyTorch MPS
> backend; CUDA paths are kept. It is not affiliated with or endorsed by the original authors. See
> **[MACOS.md](MACOS.md)** for installation, the list of changes and known limitations. The companion
> preparation fork is [knutovski/open4dgs-core](https://github.com/knutovski/open4dgs-core/tree/macos-port).
> The original README follows unchanged.

<div align="center">

# Open4DGS-Trigs

### Temporal Rigid-Body Motion for Scalable 4D Gaussian Splatting

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](LICENSE)

**Open-source implementation of the TRiGS paper — coupled SE(3) motion,
hierarchical Bézier residuals, and local anchor-centered deformation for
dynamic 3D Gaussian Splatting.**

</div>

---

> **TRiGS: Temporal Rigid-Body Motion for Scalable 4D Gaussian Splatting.**
> Suwoong Yeom\*, Joonsik Nam\*, Seunggyu Choi\*, Lucas Yunkyu Lee,
> Sangmin Kim, Jaesik Park, Joonsoo Kim, Kugjin Yun, Kyeongbo Kong†, Sukju Kang†.
> *(Sogang University · Pusan National University · POSTECH · Seoul National University · ETRI)*
> \* equal contribution &nbsp;&nbsp;&nbsp; † corresponding authors
>
> **[Project Page](https://wwwjjn.github.io/TRiGS-project_page/)** · **[Paper (arXiv)](https://arxiv.org/pdf/2604.00538)**

This trainer is based on [OpsiClear-4DGS/TriGSVanilla](https://github.com/OpsiClear-4DGS/TriGSVanilla), an independent implementation of the TRiGS paper. It modifies and extends that project for this reconstruction pipeline. Credit goes to the TriGSVanilla contributors for their implementation and to the TRiGS authors for the method; see [Citations and credits](#citations-and-credits).

One extension is **learned per-camera timing offsets**: the trainer can refine
residual sub-frame synchronization errors jointly with scene reconstruction.
This is opt-in; see [Learn camera timing offsets](#learn-camera-timing-offsets).

---

## Release status and support

This is an **initial research release**. Interfaces, defaults, and file formats may
change as the project develops. The roadmap describes planned work without
committed delivery dates.

The main Linux setup was exercised on **Ubuntu 24.04, Python 3.11, CUDA 12.8,
and an NVIDIA RTX 3090**. The trainer setup also documents a reported Windows
11 / RTX PRO 6000 Blackwell configuration. These are specific tested or reported
configurations, not a guarantee for every GPU or operating system. **WSL 2 has
not yet been validated for this release.** See the setup guides for details.

The split repositories have passed local dependency coverage checks and the
trainer package builds successfully. A fresh end-to-end GPU run using only the
split repositories has not yet been completed. Reconstruction quality, memory
requirements, and runtime depend on the capture, masks, calibration, and GPU;
large scenes may need substantial system RAM and GPU memory.

Please use GitHub issues for reproducible bugs and installation problems. Include
both repository versions, your environment, the command, and a redacted error
log. Support is best effort; response times and feature delivery are not
guaranteed. For suspected security problems, see [SECURITY.md](SECURITY.md).

## Example datasets

[Open4DGS-Datasets on Hugging Face](https://huggingface.co/datasets/asu-nem-labs/open4dgs-datasets)
hosts two example captures: **example capture** (dance, 27 cameras) and
**spidermen-01** (two performers in Spider-Man costumes, 45 cameras).
Both include original camera videos with embedded timecode and separate
alignment footage. Follow the dataset README and the
[core preparation pipeline](https://github.com/ASU-NEM-Labs/open4dgs-core#pipeline)
to extract frames, align cameras, and prepare training inputs. Downloads
preserve the camera folder structure.

The datasets use a separate license permitting **private, noncommercial
evaluation and testing only**; this repository's AGPL license does not apply
to them. Private scene-reconstruction
training for evaluation is permitted. Sign in to Hugging Face and accept the
dataset terms to obtain access. Sharing footage, renders, or reconstructed
models requires prior written permission. See the dataset license for the
full terms.

## Overview

TRiGS reconstructs dynamic 3D Gaussian scenes by assigning each primitive a
continuous, geometrically-consistent rigid-body trajectory. Unlike prior 4DGS
methods that rely on piecewise-linear velocity and short temporal windows
(and therefore fragment motion into many short-lived primitives), TRiGS keeps
a fixed canonical set whose identity persists across the whole sequence.

**Key ideas (from the paper):**

1. **Coupled SE(3) motion.** Per-primitive translation and rotation are
   optimized jointly as a single Lie-algebra coefficient
   `ζᵢ(t) = (ωᵢ, νᵢ) ∈ 𝔰𝔢(3)` and mapped to SE(3) via the closed-form
   exponential — removing the rotation/translation ambiguity that plagues
   independent parameterizations.

2. **Hierarchical Bézier residuals.** `ζᵢ(t) = ζᵢᵇᵃˢᵉ + ζᵢʳᵉˢ(τ)` where
   `ζᵢʳᵉˢ` is a quadratic Bézier curve in a normalized local time
   `τ ∈ [0, 1]` defined over the effective visibility window
   `Wᵢ = [μₜ,ᵢ − 2sₜ,ᵢ, μₜ,ᵢ + 2sₜ,ᵢ]`. This captures smooth nonlinear motion
   without over-parameterizing weakly-supervised time regions.

3. **Local anchor-centered deformation with gauge fixing.** Each primitive
   carries a learnable anchor `aᵢ` that acts as its pivot; we canonicalize it
   as `aᵢ,⊥` (orthogonal to the rotation axis) and keep only the
   axis-aligned translation `νᵢ,∥`, eliminating the optimization ambiguity
   between anchor and translation.

4. **Motion-guided relocation.** Instead of densifying, the paper recycles
   low-opacity primitives by cloning active ones weighted by a four-term score
   (opacity · motion-magnitude × visibility · accumulated ∇₂D · accumulated
   ∂L/∂μₜ). The canonical count stays fixed.

---

## Repository Structure

```
TriGS/
├── src/trigs/
│   ├── math/          # SE(3) exp/log, quaternion utilities
│   │   ├── se3.py
│   │   └── quaternion.py
│   ├── core/          # Trainer, losses, camera, renderer, transforms
│   │   ├── trainer.py       # TrigsTrainer + TrigsTrainingConfig
│   │   ├── losses.py        # reconstruction / bezier_motion / rigid_coherence
│   │   ├── camera.py
│   │   ├── renderer.py
│   │   └── transforms.py
│   ├── strategy/      # Motion-guided relocation strategy (gsplat Strategy subclass)
│   │   └── motion_guided_relocation.py
│   ├── io/            # COLMAP/SelfCap IO, normalization, trajectories, dataset, PLY
│   ├── init/          # Canonical-set builders for all input modes
│   │   ├── from_plys.py     # PLY-sequence initialization helper
│   │   ├── from_colmap.py   # ROMA/COLMAP NPZ → canonical params
│   │   ├── from_selfcap.py  # SelfCap pcd sequence → canonical params
│   │   └── roma_combine.py  # Per-frame NPYs → combined NPZ
│   └── cli/
│       ├── train_from_plys.py    # trigs-train-plys
│       ├── train_from_colmap.py  # trigs-train-colmap
│       └── train_from_selfcap.py # trigs-train-selfcap
├── scripts/
│   ├── preprocess_selfcap.py
│   ├── run_plys.sh
│   └── run_colmap.sh
├── tests/             # SE(3) math + trainer smoke tests
├── SETUP.md           # Linux CUDA and environment setup
├── LICENSE                # AGPL-3.0
├── README.md
└── pyproject.toml
```

---

## Installation

For frame extraction, calibration, masks, and initialization, use
[Open4DGS-Core](https://github.com/ASU-NEM-Labs/open4dgs-core). The
[combined setup guide](https://github.com/ASU-NEM-Labs/open4dgs-core/blob/main/SETUP.md)
explains how both repositories share one environment. The Python import name
and command prefix remain `trigs`.

```bash
git clone https://github.com/ASU-NEM-Labs/open4dgs-trigs.git
cd open4dgs-trigs
```

This project uses [uv](https://github.com/astral-sh/uv) for dependency management. On Ubuntu with an NVIDIA GPU, follow the [Linux setup guide](SETUP.md) to install Python, CUDA, PyTorch, and build tools. Then run from the Open4DGS-Trigs project directory:

```bash
uv sync --locked --preview-features extra-build-dependencies
```

The `--preview-features extra-build-dependencies` flag supplies build dependencies for `gsplat` and PyTorch3D. If using the Conda environment in the setup guide, also use `--inexact` and set `UV_PROJECT_ENVIRONMENT` as shown there.

To avoid typing the flag every time:

```bash
# bash / zsh
export UV_PREVIEW_FEATURES=extra-build-dependencies

# PowerShell (persistent)
setx UV_PREVIEW_FEATURES extra-build-dependencies
```

### Windows

See the [Windows installation instructions](SETUP.md#windows-11-with-nvidia-gpus) for a reported Blackwell build, compiler requirements, long-path troubleshooting, and the UTF-8 console setting.

The usage examples below use `uv run --no-sync` so they do not remove preparation packages installed separately in a shared environment. Install the project first; see [sharing an environment](SETUP.md#sharing-an-environment-with-preparation-tools).

---

## Usage

TriGS supports three input modes through a single trainer. All produce the
same canonical `ParameterDict` (means, quats, scales, opacities, SH, temporal
window params, anchors, ζ-base + Bézier controls) that drives training.

### 1. PLY-sequence mode

Input: a directory of dense per-frame Gaussian PLYs.
Every `keyframe_stride`-th PLY becomes part of the canonical set; the trainer
renders GT from the PLYs themselves.

```bash
uv run --no-sync trigs-train-plys \
    --input-dir /path/to/dense_frames \
    --output-dir /path/to/output \
    --config.keyframe-stride 3 \
    --config.fps 30 \
    --config.n-steps 30000
```

To reproduce the current TriGSVanilla PLY defaults without this fork's later
temporal residency, clamp, mask, and relocation changes:

```bash
uv run --no-sync trigs-train-plys \
    --input-dir /path/to/dense_frames \
    --output-dir /path/to/output \
    --device cuda \
    --vanilla-profile \
    --config.fps 30 \
    --config.keyframe-stride 3 \
    --config.n-steps 30000
```

`--vanilla-profile` was audited against
[`OpsiClear-4DGS/TriGSVanilla@bb51ea6`](https://github.com/OpsiClear-4DGS/TriGSVanilla/commit/bb51ea6482513bb3075a11e959a2beec46f1702b).
It is confined to `trigs-train-plys` and is mutually exclusive with
`--compact-profile`. It restores joint optimization from step zero, raw
unbounded temporal Gaussians, all-row rendering, Adam `beta1=0.9` for time and
duration, and verbatim relocation cloning when used with base config defaults.
It does not overwrite explicit `unfreeze_temporal_step`, `masked_rgb_loss`,
`lambda_sigma_floor`, or `sigma_floor_target` values. This permits the
upstream-documented two-phase and segmented-dynamic safeguards while retaining
an unbounded upper σ_t. The profile aligns exposed mechanics while retaining
this fork's compatibility loader, cache, diagnostics, and hardened trainer;
run the upstream checkout itself for a code-identical control.

For a segmented PLY sequence intended for clean compositing, retain the
Vanilla mechanics but enable a motion-only phase, alpha supervision, and the
one-sided σ_t floor documented by the upstream config:

```bash
uv run --no-sync trigs-train-plys \
    --input-dir /path/to/dense_frames \
    --output-dir /path/to/output \
    --device cuda \
    --vanilla-profile \
    --config.fps 30 \
    --config.keyframe-stride 3 \
    --config.n-steps 30000 \
    --config.unfreeze-temporal-step 3000 \
    --config.masked-rgb-loss \
    --config.lambda-alpha-outside-mask 1.0 \
    --config.lambda-alpha-mask 0.05 \
    --config.lambda-sigma-floor 0.1 \
    --config.sigma-floor-target 0.1
```

The floor prevents short-duration collapse but does not cap long-lived splats;
no `max_sigma_t` value is set by this recipe.

For a long dense-PLY sequence, use the opt-in input-derived profile:

```bash
uv run --no-sync trigs-train-plys \
    --input-dir /path/to/dense_frames \
    --output-dir /path/to/output \
    --device cuda \
    --compact-profile \
    --config.fps 30 \
    --config.keyframe-stride 3 \
    --config.n-steps 100000 \
    --camera.longitude-steps 12 \
    --camera.latitude-steps 10 \
    --no-render-debug \
    --no-save-plys
```

This profile is confined to `trigs-train-plys`. By default it spatially
retains 40% of the rows in the actual source keyframes (about 8.33M for a
20.8M-row canonical source). At the initial temporal width, neighboring
keyframes contribute about 2.5 cohorts' worth of Gaussian opacity mass, so 40%
starts close to one original PLY frame's effective spatial density. The
profile preserves the source PLY opacity and appearance, enables the
motion-only warm-up and local twist initialization, keeps temporal support
local, and prevents learned scales from growing far beyond the source PLY
scale distribution. It also uses the PLY-derived alpha masks for the subject
loss and outside-alpha penalty, holds the opacity ceiling through the full
run, and retains the original 12-by-10 (120-view) camera orbit. It trains the
full sequence globally; do not add temporal chunk flags unless explicitly
testing chunked training.

Use `--target-retention 0.3` when memory is tighter, or provide an exact
`--config.target-count`. Explicit target, temporal, opacity, and scale-limit
arguments override the profile defaults. Camera arguments use the normal
camera configuration directly; the profile does not replace them.

Or via the script wrapper:

```bash
bash scripts/run_plys.sh /path/to/dense_frames /path/to/output 3 30000 0
```

Outputs: `trigs_ckpt.pt`, `frames_ply/frame_000000.ply`, …, and optionally
`renders/cam_XXXX/render_YYYYYY.png`.

### 2. Multi-view COLMAP + ROMA mode

Input: COLMAP sparse reconstruction (`<data_dir>/sparse/0/` +
`<data_dir>/images/cam##/`) and either a prepared keyframes NPZ or a
directory of per-frame RoMa-triangulated point NPY files.

For the per-frame NPY input mode, the pipeline (1) runs
[`trigs.init.roma_combine`](src/trigs/init/roma_combine.py) to build an NPZ
of keyframe points with kNN-estimated velocities, (2) converts it to a TriGS
canonical set via
[`trigs.init.from_colmap.build_canonical_from_npz`](src/trigs/init/from_colmap.py)
(initializing scales from a per-keyframe kNN distance, and running the same
radius-ball anchor + (r × v) ω initialization as the PLY path), and (3)
calls `TrigsTrainer.train_from_params` with GT images loaded via
`FreeTimeParser`/`FreeTimeDataset`.

With a keyframes NPZ from `prep/init/prep-roma-trigs-memory.py` (the production path; see
[docs/run-pipeline.md](https://github.com/ASU-NEM-Labs/open4dgs-core/blob/main/docs/run-pipeline.md) for the whole pipeline), the frame range, keyframe step,
fps and splat budget come from the NPZ:

```bash
uv run --no-sync trigs-train-colmap \
    --data-dir /path/to/colmap_data \
    --output-dir /path/to/output \
    --npz-path /path/to/keyframes.npz \
    --timing-json /path/to/timing_candidate.json \
    --config.n-steps 30000
```

Omit `--timing-json` for genlocked, global-shutter footage. The prepared NPZ
records the frame range, keyframe stride, fps and target population; the
trainer resolves these from the NPZ. The normal budget defaults to 30,000
steps. Fast-frame sampling may add steps for short-lived splats; set
`--config.frame-sampling-fast-visits 0` to disable the top-up. The checkpoint
is written to the output directory as `trigs_ckpt.pt`.

Evaluate a completed run and export PLYs with:

```bash
uv run --no-sync trigs-eval-colmap --data-dir /path/to/colmap_data \
    --runs /path/to/output --step -1 --out-dir /path/to/output/eval \
    --shutter captured
uv run --no-sync trigs-export-plys --ckpt-path /path/to/output/trigs_ckpt.pt \
    --output-dir /path/to/output_plys --n-frames <FRAME_COUNT> --fps <FPS>
```

Use the capture's actual frame count and frame rate. Some final COLMAP
checkpoints do not include the training configuration. If the run overrides
deformation settings, export from a snapshot saved with
`--config.checkpoint-dir` to retain them. Evaluation uses saved viewer
snapshots for the run configuration.

From per-frame ROMA NPY files instead (builds `<output>/keyframes.npz` first):

```bash
uv run --no-sync trigs-train-colmap \
    --data-dir /path/to/colmap_data \
    --roma-input-dir /path/to/per_frame_npys \
    --output-dir /path/to/output \
    --start-frame 0 --end-frame 61 --keyframe-step 5
```

Or:

```bash
bash scripts/run_colmap.sh \
    /path/to/per_frame_npys /path/to/colmap_data /path/to/output \
    0 61 5 0
```

#### Learn camera timing offsets

For imperfectly synchronized cameras, the trainer can optimize one constant
time correction per training camera using the image reconstruction loss.
The correction is added to that camera's existing capture times, allowing
sub-frame refinement alongside the scene's motion parameters. These are
**temporal offsets**, measured in seconds; they do not change camera poses.

Enable refinement on the COLMAP training command with:

```bash
uv run --no-sync trigs-train-colmap \
    --data-dir /path/to/colmap_data \
    --output-dir /path/to/output \
    --npz-path /path/to/keyframes.npz \
    --timing-json /path/to/timing_candidate.json \
    --config.n-steps 30000 \
    --config.learn-camera-time-offsets
```

Start from the preparation pipeline's synchronization estimate. The COLMAP
loader uses `frame_timestamps.json` when available, otherwise an explicit
`--timing-json`, otherwise `sync.json`, and finally nominal frame times.
Do not combine `frame_timestamps.json` with `--timing-json`; both define the
capture-time baseline and the loader rejects that combination.

| Setting | Default | Purpose |
| --- | --- | --- |
| `--config.learn-camera-time-offsets` | Disabled | Enable temporal refinement. |
| `--config.camera-time-offset-warmup-steps` | `8000` | Let the scene settle before learning offsets. |
| `--config.lr-camera-time-offsets` | `0.0001` | Offset optimizer learning rate. |
| `--config.lambda-camera-time-offset-reg` | `0.01` | Regularize corrections toward zero, preserving the initial timing estimate. |
| `--config.camera-time-offset-clamp-s` | `0.0083` | Raw-parameter clamp scale in seconds, about half a frame at 60 fps. |

The trainer clamps the raw offsets and then centers the effective corrections
to zero mean, so it learns relative camera timing rather than a global shift
of the sequence. The clamp value is not a strict bound on the corrections
after centering. Runs that do not pass the warmup will not learn offsets.

Optionally, `--config.camera-time-offset-observability-json /path/to/observability.json`
uses the preparation tools' per-camera observability estimates to reduce
corrections for cameras with weak timing evidence. Without it, all cameras
use a scale of one.

This feature refines small residual timing errors; it does not model clock
drift across a sequence or the within-image timing variation of rolling
shutter. Keep the appropriate rolling-shutter model enabled separately.
Weak motion or inaccurate scene geometry can make timing difficult to infer,
so compare reconstruction quality against a run with fixed timing.

### 3. SelfCap (4DV) mode

Input: a SelfCap `<scene>-release` directory with `videos/`, `optimized/`,
and per-frame `pcds/`.
Dataset/project page: [Long Volumetric Video (SelfCap Dataset)](https://zju3dv.github.io/longvolcap/).

First preprocess a frame window (extract jpgs + copy matching point clouds +
write `preprocess_meta.json`):

```bash
uv run --no-sync python scripts/preprocess_selfcap.py \
    --scene-dir /path/to/corgi-release \
    --output-dir /path/to/corgi_prepped \
    --start-frame 0 --end-frame 1000 \
    --factor 1 --fps 60 --workers 24
```

Then train TriGS on the preprocessed directory:

```bash
uv run --no-sync trigs-train-selfcap \
    --data-dir /path/to/corgi_prepped \
    --output-dir /path/to/output \
    --dense-subdir dense_pcds \
    --sparse-subdir pcds \
    --config.keyframe-stride 3 \
    --config.n-steps 30000
```

Notes:
- `preprocess_selfcap.py` auto-detects `pcd` absolute-frame origin from the
  minimum filename in `pcds/`; override with `--pcd-origin` when needed.
- By default, `preprocess_selfcap.py` rectifies images and rewrites
  `optimized/intri.yml` to a pinhole model baked to cropped image space.
  Pass `--no-rectify-pinhole` to keep original distorted images/calibration.
- Pass `--no-sync` to `trigs-train-selfcap` to disable `optimized/sync.json`
  per-camera timing correction.

## Interactive trainer viewer

The [trainer viewer](src/trigs/core/viewer.py) renders trained Gaussian scenes
in a browser using `viser`, `nerfview`, and the CUDA `gsplat` renderer. It opens
`.pt` checkpoints and training snapshots. To edit exported `.trigsbin` files,
use the [customized SuperSplat editor](https://cmlapps.com/trigs-editor/).

### Open a checkpoint

Install the trainer environment first, activate it, and run from the
`open4dgs-trigs` checkout root:

```bash
python -m trigs.core.viewer \
  --ckpt /path/to/run/trigs_ckpt.pt \
  --port 8080 --device cuda
```

Open **http://localhost:8080** in your browser. The Python process needs an
NVIDIA CUDA GPU; the browser can run on a different machine. Keep the process
running while viewing and press `Ctrl+C` in its terminal to stop it. With the
package installed, the module command works from any directory. From the
checkout, `python src/trigs/core/viewer.py` accepts the same options.

Use **Frame** to scrub through time, **Auto Play** and **Play Speed (FPS)** for
playback, and the viewport controls to move the camera. The UI also provides
background color, static/dynamic visibility, opacity and scale controls, and
diagnostic color modes for motion, duration, and opacity. Diagnostic controls
change the display; they do not rewrite the checkpoint.

Useful options:

| Option | Purpose |
| --- | --- |
| `--fps 60 --total-frames 300` | Override playback metadata with your capture's values. Without metadata or overrides, the viewer falls back to 30 fps and 300 frames. |
| `--t-start <SECONDS> --t-end <SECONDS>` | Override the displayed time range. |
| `--temporal-threshold <VALUE>` | Match the run's `temporal_active_threshold` when comparing the viewer with training. The viewer CLI defaults to `0.05`; the current trainer default is `0.01`. |
| `--no-spatial-filter` | Disable the display's spatial outlier filter if expected content is missing. |
| `--no-precompute` | Disable the per-frame visibility cache to reduce memory use. |
| `--device cuda:1` | Use a second GPU when available. |
| `--npz-path /path/to/keyframes.npz` | Enable source-cohort diagnostics when the original NPZ remains row-aligned with the model; unnecessary for normal viewing. |

Use the settings from the actual run, particularly for older checkpoints or
runs with custom defaults. For all options, run
`python -m trigs.core.viewer --help`.

### Follow training as snapshots arrive

Run the viewer in a separate terminal from training. For example, from the
installed trainer environment:

```bash
trigs-train-colmap --data-dir /path/to/capture \
  --npz-path /path/to/capture/keyframes.npz --output-dir /path/to/run \
  --config.viewer-snapshot-every 1000 --config.viewer-snapshot-keep-last 3
```

```bash
python -m trigs.core.viewer \
  --watch /path/to/run/viewer_snapshots --port 8080 --device cuda
```

The viewer follows completed snapshots, preserving the camera and UI state;
this is periodic snapshot viewing rather than a render of every training step.
If there are no snapshots yet, it waits for the first one **before starting the
browser server**. Snapshot output defaults to every 1,000 steps with the last
three retained; `--config.viewer-snapshot-every 0` disables it. Use a second GPU
when possible, since training and viewing otherwise share GPU memory.

- Snapshots go to `<output_dir>/viewer_snapshots/` (override with
  `--config.viewer-snapshot-dir`); `latest.json` is replaced atomically after
  each file. `--config.viewer-snapshot-keep-every N` also keeps every Nth step
  as a milestone, and `--config.viewer-snapshot-dtype float16` halves `shN`.
- Chunked runs (`temporal_chunk_len_s > 0`) write the whole CPU master
  between chunk calls, so a snapshot is always a complete model. In round 1,
  windows after the last-trained one are still at their initial state; the
  "Jump to last-trained window" button goes to the part that just changed.
- `--watch` also follows a `--config.checkpoint-dir` (safety checkpoints);
  `--no-auto-reload` pins the first snapshot and reloads only on request, and
  the "Snapshot" dropdown steps back to an older one.
- The viewer builds the new model next to the old one and swaps it in, so it
  briefly holds two copies. If memory is tight it frees the old one first;
  `--no-precompute` also drops the per-frame visibility cache.
- Snapshot cost is logged to TensorBoard (`viewer_snapshot/save_s`,
  `viewer_snapshot/bytes`).


### View on a remote GPU machine

Run the viewer on the GPU machine. From a terminal on your local computer,
forward its port and leave the SSH connection open:

```bash
ssh -N -L 8080:localhost:8080 user@gpu-host
```

Then open **http://localhost:8080** locally. Use trusted machines and protect
the server ports from public access; see [SECURITY.md](SECURITY.md).

### Inspect initialization NPZ files

The point-cloud NPZ viewer lives in **Open4DGS-Core**, at
`prep/init/npz-viewer.py`. It shows the initialization before training, including
velocity overlays and stage/detail-region previews, and does not require CUDA.
See the [NPZ viewer instructions](https://github.com/ASU-NEM-Labs/open4dgs-core#npz-initialization-viewer).

---

## Key Concepts

### The canonical set is fixed

Unlike methods that densify/prune during training, TriGS keeps the canonical
primitive count constant. Capacity that ends up unused (`σ(αᵢ) < τ_α`) is
recycled by **motion-guided relocation** — cloning active primitives into
inactive slots, weighted by four normalized cues:

```
score_i = p^(α) + p^(dν) + p^(2D) + p^(t)
```

where the four terms are opacity, motion × visibility, accumulated 2D
projected-means gradient, and accumulated ∂L/∂μₜ gradient. Implemented as a
gsplat [`Strategy`](src/trigs/strategy/motion_guided_relocation.py) subclass
so it plugs into any gsplat training loop.

### Time is in physical units (seconds)

All temporal quantities (`μₜ`, `σₜ`, `ν`, `ω`, Bézier controls, loss weights)
are stored in seconds — not normalized `[0, 1]`. This makes paper
hyperparameters (`λ_motion = 1e-4`, `λ_rigid = 1`, `λ_c = 50`, …) transfer
directly across sequences of different length and FPS.

### Two-phase training

1. **Motion-only phase** (first `unfreeze_temporal_step` iters): opacities,
   `μₜ`, `log σₜ` are frozen. Forces the model to find a motion decomposition
   that fits all frames before it can cheat by shrinking temporal windows.
2. **Full training phase**: temporal params unfreeze; Adam moments are
   reset for a clean restart.

### Initialization is shared between the two input modes

Both the PLY and COLMAP paths end up running the **same** three-step init
on their per-keyframe Gaussians:

1. Centered finite-difference velocity from ±1 frame neighbors.
2. Radius-ball anchor centroids (anchor ≠ mean, so rotation actually
   displaces the primitive).
3. ω = (r × v) / ‖r‖² with residual ν = v − ω × r — so the translation
   component only carries what rotation can't explain.

---

## Development

```bash
# Install with dev dependencies
uv sync --preview-features extra-build-dependencies --group dev

# Run tests (SE(3) math + trainer smoke tests)
uv run --no-sync pytest

# Type check / lint
uv run --no-sync mypy src/trigs/ --ignore-missing-imports
uv run --no-sync ruff check src/trigs/
```

---

## Citations and credits

This trainer is derived from [TriGSVanilla](https://github.com/OpsiClear-4DGS/TriGSVanilla). The PLY control path references [upstream commit bb51ea6](https://github.com/OpsiClear-4DGS/TriGSVanilla/commit/bb51ea6482513bb3075a11e959a2beec46f1702b). See [third-party notices](THIRD_PARTY_NOTICES.md) for incorporated code and license notices.

If you use this code or build on the TRiGS method, please cite the original
paper:

```bibtex
@article{yeom2026trigs,
  title   = {TRiGS: Temporal Rigid-Body Motion for Scalable 4D Gaussian Splatting},
  author  = {Yeom, Suwoong and Nam, Joonsik and Choi, Seunggyu and
             Lee, Lucas Yunkyu and Kim, Sangmin and Park, Jaesik and
             Kim, Joonsoo and Yun, Kugjin and Kong, Kyeongbo and Kang, Sukju},
  journal = {arXiv preprint},
  year    = {2026},
  url     = {https://wwwjjn.github.io/TRiGS-project_page/}
}
```

Please also cite the projects this implementation builds on:

- **[3D Gaussian Splatting](https://arxiv.org/abs/2308.04079)** (Kerbl et al., SIGGRAPH 2023)
- **[FreeTimeGS](https://openaccess.thecvf.com/content/CVPR2025/html/Wang_FreeTimeGS_Free_Gaussian_Primitives_at_Anytime_Anywhere_for_Dynamic_Scene_CVPR_2025_paper.html)** (Wang et al., CVPR 2025) — the translation-only baseline
  TriGS generalizes, and the dataset-loading lineage of this code.
- **[Representing Long Volumetric Video with Temporal Gaussian Hierarchy](https://arxiv.org/abs/2412.09608)** (Xu et al., 2024; LongVolCap / SelfCap) — temporal hierarchy inspiration and, when used, dataset credit. The hierarchy in this trainer is adapted and optional.
- **[gsplat](https://arxiv.org/abs/2409.06765)** (Ye et al., 2024) — CUDA rasterization backend.
- **[COLMAP](https://colmap.github.io/#citation)** — camera and reconstruction tooling; see its citation instructions for the papers relevant to the features used.

---

## License

The trainer is AGPL-3.0. See [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The MultiNeRF-derived trajectory code retains its Apache-2.0 notice and [license](LICENSE-MULTINERF); the COLMAP-derived reader retains its own source notice.

This project is an independent open-source re-implementation of the TRiGS
paper. It is not affiliated with, nor endorsed by, the paper's authors or
institutions.
