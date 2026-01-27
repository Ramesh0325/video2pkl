# PARE: Video to 3D Human Motion (PKL)

Convert monocular video into 3D human motion PKL files using **PARE** (Part Attention Regressor) with optional motion smoothing via **SmoothNet**. This repository is set up for **inference and analysis**—not for training from scratch.

---

## 1. Project Overview

### What does this repo do?

The pipeline takes a **video** (e.g. `dance.mp4`) and produces a **PKL file** that contains 3D human pose and shape over time in SMPL format.

**“Video → PKL”** means:

- **Input:** A video file (MP4, etc.) with visible people.
- **Output:** A Python pickle/joblib file (`.pkl`) with, per person and per frame:
  - SMPL pose (axis-angle, 72D) and global orientation
  - Root translation (3D)
  - 3D joint positions, shape parameters, and frame indices

That PKL can then be used for motion analysis, retargeting to character rigs (e.g. in Maya), or as input to other research/creative tools.

### Typical use cases

- **Motion analysis** — Extract 3D pose from video for biomechanics or movement studies.
- **Animation** — Feed motion into DCC tools (e.g. Maya) for retargeting or editing.
- **Research** — Use as a preprocessing step for action recognition, re-enactment, or similar.
- **Validation** — Check that generated motion is smooth and suitable for downstream use (via built-in validators).

---

## 2. Features

- **PARE-based 3D pose estimation** — Predicts SMPL body parameters and 3D joints from video using the PARE model (ICCV 2021).
- **Video-to-PKL inference** — Single entry point: video in, SMPL PKL out (multi-person supported via tracking).
- **Motion validation and diagnostics** — Validate PKL structure and motion quality (jitter, angular velocity, root stability).
- **Motion smoothing with SmoothNet** — Optional temporal smoothing of 3D joints to reduce jitter; SmoothNet is **vendored** in this repo (not a git submodule).

---

## 3. Repository Structure

| Path | Purpose |
|------|---------|
| **`pare/`** | PARE model, backbones, and inference utilities. Core pose estimation lives here. |
| **`smoothnet/`** | SmoothNet code and configs, vendored in-repo. Used for temporal smoothing of 3D joints. |
| **`scripts/`** | Install scripts (`install_conda.sh`, `install_pip.sh`), data prep (`prepare_data.sh`), and other helpers (demo, eval, occlusion analysis). |
| **`data/`** | Created by `prepare_data.sh`. Holds PARE checkpoints, SMPL model, and other assets (see Checkpoints). |
| **`docs/`** | Assets (e.g. GIFs) and documentation. |

### Main scripts (root directory)

| Script | Role |
|--------|------|
| **`infer.py`** | Main inference script: video → SMPL PKL. Supports `--enable_smoothnet` and related options. |
| **`run_inference.sh`** | Wrapper that sets up the environment and runs inference + validation + optional SmoothNet/motion stabilizer. Easiest way to run “video → PKL.” |
| **`validate_pkl.py`** | Validates PKL *structure* (shapes, keys, no NaNs). |
| **`validate_pkl_motion.py`** | Validates that root motion exists and is non-trivial (basic “does this PKL have motion?” check). |
| **`pkl_motion_validator.py`** | Motion *quality* checks: angular velocity, root jitter, bad-frame ratio. Reports PASS / PASS_WITH_CLEANUP / FAIL. |
| **`motion_stabilizer.py`** | Post-processing stabilizer: quaternion smoothing, outlier replacement, root smoothing. Used after SmoothNet in the pipeline. |
| **`analyze_motion_type.py`** | Analyzes whether motion is progressing, looping, or oscillating (useful for diagnostics). |
| **`diagnose_pkl.py`** | PKL diagnostics (contents, shapes, basic stats). |

---

## 4. Installation

### Python

- **Recommended:** Python **3.7–3.9** (tested with conda on Ubuntu).

### Environment (conda, recommended)

```bash
# Create and activate env
conda create -n PARE python=3.8 -y
conda activate PARE
```

Or use the project scripts:

```bash
# Conda-based install
source scripts/install_conda.sh

# Or pip/venv
source scripts/install_pip.sh
```

### Dependencies

For **inference only** (video → PKL):

```bash
pip install -r requirements_inference.txt
```

For the full repo (e.g. demos, evaluation, training-related code):

```bash
pip install -r requirements.txt
```

### External tools

- **ffmpeg** — Used to decode video and extract frames. Install system-wide (e.g. `apt install ffmpeg` on Ubuntu).
- **CUDA** (optional) — Speed up inference. CPU works but is slower.

---

## 5. Checkpoints

**Checkpoints** are pre-trained model weights. The pipeline uses two kinds:

1. **PARE checkpoints** — For 3D pose/shape from images.  
   - Download and unpack via: `source scripts/prepare_data.sh` (downloads ~1.3 GB from Google Drive, creates `data/`).  
   - After that, PARE assets live under **`data/pare/checkpoints/`** (e.g. `pare_checkpoint.ckpt`, `pare_config.yaml`).  
   - Total size is on the order of **~1–2 GB** (PARE + SMPL + YOLO, etc.).

2. **SmoothNet checkpoints** — For temporal smoothing of 3D joints.  
   - **Not** included in the repo by default.  
   - Download from the [SmoothNet drive](https://drive.google.com/drive/folders/19Cu-_gqylFZAOTmHXzK52C80DKb0Tfx_?usp=sharing) (see `smoothnet/DOWNLOAD_CHECKPOINTS.md`).  
   - Place them under **`smoothnet/data/checkpoints/pw3d_spin_3D/`**, e.g.:
     - `checkpoint_8.pth.tar`
     - or `checkpoint_32.pth.tar` (adjust `--smoothnet_checkpoint` if you use a different file).

If you skip SmoothNet checkpoints, you can still run inference without `--enable_smoothnet`; only the smoothing step will be disabled.

---

## 6. Running Inference

### Quick run (recommended)

Use the shell script so that environment, paths, and optional SmoothNet are handled for you:

```bash
./run_inference.sh inputs/your_video.mp4
```

You can pass the video path as the first argument or let the script prompt you.

### What the script does

1. Activates the conda env (e.g. `PARE`).
2. Runs **PARE inference** (`infer.py`) on the video.
3. Optionally runs **SmoothNet** and **motion stabilizer** (if enabled and checkpoints are present).
4. Validates the PKL and motion quality.
5. Writes outputs under **`outputs/<video_basename>_<timestamp>/`** and copies the final PKL to **`output-pkl-files/<video_basename>.pkl`**.

### Direct use of `infer.py`

```bash
python infer.py \
  --video inputs/your_video.mp4 \
  --out outputs/my_run \
  --enable_smoothnet \
  --smoothnet_checkpoint smoothnet/data/checkpoints/pw3d_spin_3D/checkpoint_8.pth.tar \
  --smoothnet_window_size 8
```

- **Input:** `--video` = path to a single video file.
- **Output:** under `--out` you get (among others) `smpl_output.pkl` containing SMPL poses, root translation, joints, etc., keyed by person ID.

### Inputs and outputs

- **Inputs:**  
  - Video path.  
  - Optionally SmoothNet checkpoint path and window size.

- **Outputs:**  
  - **`<out>/smpl_output.pkl`** — SMPL parameters and 3D joints per person and per frame.  
  - **`output-pkl-files/<video_basename>.pkl`** — Copy of the final PKL for easy reuse.  
  - **`<out>/motion_validation.json`** — Motion quality report (if validation is run).  
  - **`<out>/bad_frames.csv`** — Frames flagged as problematic by the motion validator (if generated).

---

## 7. Motion Validation & Analysis

### Validating a PKL

**Structure check** (required keys, shapes, no NaNs):

```bash
python validate_pkl.py --pkl path/to/smpl_output.pkl
```

**Basic motion check** (root translation has meaningful motion):

```bash
python validate_pkl_motion.py path/to/smpl_output.pkl
```

**Motion quality** (jitter, angular velocity, bad-frame ratio, PASS/PASS_WITH_CLEANUP/FAIL):

```bash
python pkl_motion_validator.py --pkl path/to/smpl_output.pkl --fps 30 --output report.json --csv bad_frames.csv
```

- Uses thresholds for angular velocity and root displacement.  
- Writes a JSON report and an optional CSV of bad frame indices.  
- See **`PKL_MOTION_VALIDATOR_README.md`** for thresholds and interpretation.

### Understanding the validators

- **`validate_pkl.py`** — Sanity check that the PKL is loadable and has expected structure.  
- **`validate_pkl_motion.py`** — Ensures the PKL is “dynamic” enough for use (e.g. root not constant). Run: `python validate_pkl_motion.py <pkl_path>`.  
- **`pkl_motion_validator.py`** — Production-oriented quality: “Is this motion smooth enough for rigging/retargeting?”

---

## 8. Motion Smoothing (SmoothNet)

### What SmoothNet does

SmoothNet is a small temporal network that smooths sequences of 3D joint positions in time. It reduces high-frequency jitter and implausible spikes while keeping the overall motion. In this repo it runs on the **joints** produced by PARE; pose/translations are then refined for consistency.

### How it fits in the pipeline

When you use **`run_inference.sh`** or **`infer.py --enable_smoothnet`**:

1. PARE predicts poses and 3D joints per frame.
2. SmoothNet smooths the 3D joints (configurable window, e.g. 8 or 32).
3. Root translation and poses are filtered/updated to match the smoothed joints.
4. Optionally **`motion_stabilizer.py`** runs (outlier replacement, quaternion smoothing, root clamping).

SmoothNet is **optional**. If you do not download its checkpoints or do not pass `--enable_smoothnet`, inference still runs; only the SmoothNet and stabilizer steps are skipped or no-ops.

### Usage

- **Via script:** `./run_inference.sh inputs/video.mp4` (SmoothNet is enabled by default in the script if the checkpoint exists).
- **Via `infer.py`:**  
  - `--enable_smoothnet`  
  - `--smoothnet_checkpoint smoothnet/data/checkpoints/pw3d_spin_3D/checkpoint_8.pth.tar`  
  - `--smoothnet_window_size 8` (must match the checkpoint, e.g. 8 for `checkpoint_8.pth.tar`).

See **`smoothnet/DOWNLOAD_CHECKPOINTS.md`** and **`SMOOTHNET_FIXES.md`** for checkpoint paths and known fixes.

---

## 9. Notes & Limitations

- **GPU** — Strongly recommended for PARE inference. CPU is supported but slow.
- **Inference only** — This fork is aimed at **inference and analysis**. Training scripts may exist under `scripts/` but are not the main focus; training from scratch is not documented here.
- **Failure cases** — Performance can degrade with:
  - Heavy occlusions.
  - Very fast motion or motion blur.
  - Multiple people with similar appearance (tracking can mix identities).
  - Unusual poses or camera angles.
- **SmoothNet** — Checkpoint **window size** (e.g. 8 vs 32) is fixed by the model; use the value that matches the file (e.g. `checkpoint_8.pth.tar` → window 8).

---

## 10. License & Credits

- **PARE** — [PARE: Part Attention Regressor for 3D Human Body Estimation (ICCV 2021)](https://arxiv.org/abs/2104.08527).  
  Kocabas et al.  
  Code and idea from the [original PARE repository](https://github.com/mkocabas/PARE).

- **SmoothNet** — SmoothNet temporal smoothing is from [cure-lab/SmoothNet](https://github.com/cure-lab/SmoothNet). It is **vendored** in this repo (under `smoothnet/`) for inference only.

This repository builds on that prior work to provide a **video → PKL** inference pipeline with validation and smoothing. For license terms, see the **LICENSE** file in this repo and any licenses under `pare/` and `smoothnet/`.
