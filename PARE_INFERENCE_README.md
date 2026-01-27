# PARE Video to SMPL PKL Inference

This directory contains the PARE inference pipeline that converts monocular video to SMPL motion PKL files.

## Quick Start

```bash
# Run inference on a video
./run_inference.sh inputs/your_video.mp4

# Output will be in: outputs/your_video_TIMESTAMP/smpl_output.pkl
```

## Files

- **`infer.py`** - Main inference script (video → SMPL PKL)
- **`run_inference.sh`** - Convenience script to run inference
- **`validate_pkl.py`** - Validate PKL file structure
- **`validate_pkl_motion.py`** - Validate PKL contains motion data

## Output Format

The generated PKL file contains:

```python
{
    1: {  # person_id
        'pose': (N, 72),           # Axis-angle rotations [global(3) + body(69)]
        'global_orient': (N, 3),   # Root rotation (redundant with first 3 of pose)
        'betas': (N, 10),          # SMPL shape parameters
        'trans': (N, 3),           # Root translation (camera-relative)
        'joints3d': (N, 49, 3),    # 3D joint positions
        'frame_ids': (N,),         # Frame indices
    }
}
```

## Key Features

- ✅ Root translation (`trans`) for hip movement in Maya
- ✅ Temporal smoothing (optional, disabled by default)
- ✅ Multi-person tracking support
- ✅ Camera movement handling

## Usage

### Basic Inference

```bash
python infer.py --video inputs/video.mp4 --out outputs/video_output
```

### With Custom Smoothing (if needed)

```bash
python infer.py --video inputs/video.mp4 --out outputs/video_output \
    --enable_pose_smoothing \
    --smoothing_min_cutoff 0.004 \
    --smoothing_beta 0.7
```

## Integration with Maya Pipeline

The PKL file is ready for import into your Maya pipeline:
- `pose`: (N, 72) axis-angle rotations
- `trans`: (N, 3) root translation for hip movement

See your existing Maya import scripts for details.

