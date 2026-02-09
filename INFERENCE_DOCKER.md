# PARE Inference Docker (GPU-only)

This document describes the **GPU-only, inference-only** Docker image for PARE. All required model weights and assets are **bundled inside the container** except the SMPL body model, which you **must mount** due to licensing.

**The only required external dependency is the SMPL model, which must be mounted.** No other downloads, prep scripts, or mounts are required to run inference on the first run.

---

## 1. Quick run

On a machine with NVIDIA GPU and Docker (with `--gpus` support):

```bash
docker run --rm --gpus all \
  -v /path/to/your/smpl:/app/data/body_models/smpl \
  -v /path/to/your/video.mp4:/input.mp4 \
  pare:gpu \
  python infer.py --video /input.mp4 --out /app/outputs/run1
```

- Replace `/path/to/your/smpl` with a directory that contains your **SMPL model files** (e.g. `SMPL_NEUTRAL.pkl`; layout must be compatible with the `smplx` library).
- Replace `/path/to/your/video.mp4` with your input video path.
- Output will be written inside the container at `/app/outputs/run1/`. To get it on the host, add a mount for the output directory (see below).

**With output directory on the host:**

```bash
docker run --rm --gpus all \
  -v /path/to/your/smpl:/app/data/body_models/smpl \
  -v /path/to/your/video.mp4:/input.mp4 \
  -v /path/to/outputs:/app/outputs \
  pare:gpu \
  python infer.py --video /input.mp4 --out /app/outputs/run1
```

Then find `smpl_output.pkl` under `/path/to/outputs/run1/` on your host.

---

## 2. Model weights and assets

Every model weight or asset used for inference is classified as follows:

- **Bundled in the container** – Already in the image; no action needed.
- **Must be mounted** – You must provide it by bind-mounting a directory or file.
- **Downloaded at runtime** – Not used; this image does **not** perform any runtime downloads.

| Asset | Status | Location in container | Notes |
|-------|--------|----------------------|--------|
| PARE checkpoint | Bundled | `/app/data/pare/checkpoints/pare_checkpoint.ckpt` | Loaded via `--ckpt` (default). |
| PARE config YAML | Bundled | `/app/data/pare/checkpoints/pare_config.yaml` | Loaded via `--cfg` (default). |
| SMPL body model | **Must be mounted** | `/app/data/body_models/smpl` | License restriction; not bundled. Mount your SMPL model directory here. |
| J_regressor_extra.npy | Bundled | `/app/data/J_regressor_extra.npy` | Used by PARE SMPL head. |
| YOLOv3 weights | Bundled | `/app/.torch/models/yolov3.weights` | Person detection; no `$HOME/.torch` download at runtime. |
| YOLOv3 config | Bundled | `/app/.torch/config/yolov3.cfg` | Model definition for YOLOv3. |
| SmoothNet checkpoint | Not bundled | N/A | Optional. To use `--enable_smoothnet`, mount a directory that contains a SmoothNet checkpoint and pass `--smoothnet_checkpoint` and `--smoothnet_window_size`. |

**Runtime downloads:** None. The image is designed so that a run with only the SMPL mount and a video works without network access.

---

## 3. Build the image

From the repository root:

```bash
docker build -t pare:gpu .
```

Build-time requirements:

- Network access (to download the PARE data zip, YOLOv3 weights/cfg if not in the zip, and pip packages).
- Sufficient disk space for the image (several GB).

The Dockerfile:

- Uses a **CUDA 11.8** base image with PyTorch.
- Installs system deps (ffmpeg, git, OpenGL libs for OpenCV).
- Installs Python deps from `requirements_inference.txt` (keeping the base PyTorch).
- Installs `yolov3-pytorch` and `multi-person-tracker` from git.
- Downloads the PARE data zip (via `gdown`), unpacks it under `/app/data`, then **removes** `data/body_models/smpl` and creates an empty `/app/data/body_models/smpl` so you can mount over it.
- Ensures YOLOv3 weights and config are under `/app/.torch/` (using `ENV HOME=/app`) so no runtime download is needed.
- Copies application code and sets the entrypoint to verify GPU and SMPL before running your command.

---

## 4. Runtime behavior

On container start, the entrypoint:

1. **Checks that CUDA is available.** If `torch.cuda.is_available()` is false, the container exits with an error (e.g. “CUDA is not available. This image is GPU-only.”). There is no CPU fallback.
2. **Checks that the SMPL directory exists and is non-empty** at `/app/data/body_models/smpl`. If it is missing or empty, the container exits with a clear message that you must mount the SMPL model directory.
3. **Runs your command** (e.g. `python infer.py --video ... --out ...`).

Errors are explicit: they state what is missing (GPU or SMPL) and how to fix it (e.g. use `--gpus all`, mount SMPL at `/app/data/body_models/smpl`).

---

## 5. Optional: SmoothNet

SmoothNet is **not** bundled. To use temporal smoothing:

1. Download a SmoothNet checkpoint (e.g. from the SmoothNet drive; see `smoothnet/DOWNLOAD_CHECKPOINTS.md`).
2. Place it in a directory on your host (e.g. `./smoothnet_checkpoints/pw3d_spin_3D/checkpoint_32.pth.tar`).
3. Mount that directory and pass the path and window size to `infer.py`:

```bash
docker run --rm --gpus all \
  -v /path/to/smpl:/app/data/body_models/smpl \
  -v /path/to/video.mp4:/input.mp4 \
  -v /path/to/outputs:/app/outputs \
  -v /path/to/smoothnet_checkpoints:/app/smoothnet/data/checkpoints \
  pare:gpu \
  python infer.py --video /input.mp4 --out /app/outputs/run1 \
    --enable_smoothnet \
    --smoothnet_checkpoint /app/smoothnet/data/checkpoints/pw3d_spin_3D/checkpoint_32.pth.tar \
    --smoothnet_window_size 32
```

---

## 6. Summary

- **GPU-only:** The container fails immediately if CUDA is not available; no CPU fallback.
- **Bundled:** PARE checkpoint, PARE config, `J_regressor_extra.npy`, YOLOv3 weights and config are inside the image at fixed paths.
- **Mounted:** Only the **SMPL model directory** is required to be mounted at `/app/data/body_models/smpl`.
- **No runtime downloads:** All required assets are either bundled or provided by your SMPL mount. Optional SmoothNet requires an additional mount and CLI args if you use it.

After building the image once, you can run inference on a fresh GPU machine with only an input video and the SMPL mount; no other downloads or prep are required.
