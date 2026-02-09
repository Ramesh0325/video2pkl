# PARE: GPU-only inference image. All weights bundled except SMPL (must be mounted).
# Build: docker build -t pare:gpu .
# Run:   docker run --rm --gpus all -v /path/to/smpl:/app/data/body_models/smpl \
#          -v /path/to/video.mp4:/input.mp4 pare:gpu \
#          python infer.py --video /input.mp4 --out /app/outputs/run1

# -------------------------------
# Base: CUDA 11.8 + Python + PyTorch (GPU)
# -------------------------------
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HOME=/app
WORKDIR /app

# -------------------------------
# System dependencies
# -------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    unzip \
    curl \
    && rm -rf /var/lib/apt/lists/*

# -------------------------------
# Python deps (keep base torch; add rest from requirements_inference)
# -------------------------------
COPY requirements_inference.txt .
RUN pip install --no-cache-dir \
    numpy "scipy>=1.5.0" \
    "opencv-python>=4.1.0" "pillow>=8.0.0" "joblib>=0.15.0" \
    "pyyaml>=5.3.0" "yacs>=0.1.8" \
    "tqdm>=4.47.0" "loguru>=0.3.0" \
    "smplx>=0.1.28" \
    gdown \
    trimesh pyrender \
    && rm -f requirements_inference.txt

# Person detection and tracking
RUN pip install --no-cache-dir \
    "git+https://github.com/mkocabas/yolov3-pytorch.git" \
    "git+https://github.com/mkocabas/multi-person-tracker.git"

# -------------------------------
# Bundled assets: PARE data zip → /app/data (SMPL excluded)
# -------------------------------
RUN mkdir -p /app/.torch/models /app/.torch/config

RUN gdown "1qIq0CBBj-O6wVc9nJXG-JDEtWPzRQ4KC" -O /app/pare-data.zip \
    && unzip -q /app/pare-data.zip -d /app \
    && rm /app/pare-data.zip \
    && (test -d /app/data || (mv /app/pare-github-data/data /app/data 2>/dev/null || true)) \
    && rm -rf /app/pare-github-data 2>/dev/null || true

# Remove SMPL from image (user must mount); keep empty mount point
RUN rm -rf /app/data/body_models/smpl && mkdir -p /app/data/body_models/smpl

# YOLOv3: bundle weights and config so no $HOME/.torch runtime download
RUN if [ -f /app/data/yolov3.weights ]; then mv /app/data/yolov3.weights /app/.torch/models/; fi
RUN curl -sSL -o /app/.torch/config/yolov3.cfg \
    "https://raw.githubusercontent.com/mkocabas/yolov3-pytorch/master/yolov3/config/yolov3.cfg"

# If zip did not contain yolov3.weights, download once at build time (no runtime download)
RUN if [ ! -f /app/.torch/models/yolov3.weights ]; then \
    curl -sSL -o /app/.torch/models/yolov3.weights "https://pjreddie.com/media/files/yolov3.weights"; \
    fi

# -------------------------------
# Application code
# -------------------------------
COPY setup.py .
COPY pare/ pare/
COPY smoothnet/ smoothnet/
COPY infer.py validate_pkl.py validate_pkl_motion.py pkl_motion_validator.py motion_stabilizer.py \
    analyze_motion_type.py diagnose_pkl.py .
COPY run_inference.sh .
COPY scripts/ scripts/
COPY entrypoint.sh /app/entrypoint.sh

RUN pip install -e . && chmod +x /app/entrypoint.sh

# -------------------------------
# Runtime: verify GPU + SMPL, then run user command
# -------------------------------
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["python", "infer.py", "--help"]
