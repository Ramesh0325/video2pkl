#!/usr/bin/env bash
# GPU-only inference: fail fast if CUDA or SMPL is missing.
set -e

echo "Checking GPU (CUDA)..."
python -c "
import torch, sys
if not torch.cuda.is_available():
    print('ERROR: CUDA is not available. This image is GPU-only.')
    print('Ensure you run with: docker run --gpus all ...')
    sys.exit(1)
print('CUDA OK:', torch.cuda.get_device_name(0))
"

echo "Checking SMPL model (required mount)..."
SMPL_DIR='/app/data/body_models/smpl'

if [ ! -d \"$SMPL_DIR\" ]; then
    echo \"ERROR: SMPL directory not found: $SMPL_DIR\"
    echo \"You must mount your SMPL model directory:\"
    echo \"  -v /path/to/your/smpl:$SMPL_DIR\"
    exit 1
fi

if [ -z \"\$(ls -A \"$SMPL_DIR\" 2>/dev/null)\" ]; then
    echo \"ERROR: SMPL directory is empty: $SMPL_DIR\"
    echo \"Mount a directory containing SMPL model files (e.g. SMPL_NEUTRAL.pkl).\"
    echo \"  -v /path/to/your/smpl:$SMPL_DIR\"
    exit 1
fi

echo "SMPL directory OK"

exec "$@"
