# PKL Motion Quality Validator

## Overview

`pkl_motion_validator.py` automatically detects jitter and motion artifacts in SMPL/PARE PKL files to determine if motion is artist-ready for Maya animation.

## Features

### 1. Joint Angular Velocity Check
- Converts axis-angle rotations to rotation angles
- Computes frame-to-frame angular deltas
- Converts to degrees/second using FPS
- **Thresholds:**
  - Warning: >120°/sec
  - Error: >250°/sec

### 2. Root Translation Jitter Check
- Computes frame-to-frame ΔX, ΔY, ΔZ of root
- Measures displacement per frame (cm)
- Checks both per-axis and magnitude
- **Thresholds:**
  - Per-axis warning: >0.3 cm/frame
  - Per-axis error: >1.0 cm/frame
  - Magnitude warning: >5.0 cm/frame
  - Magnitude error: >15.0 cm/frame

### 3. Bad Frame Ratio
- A frame is "bad" if any joint or root exceeds error threshold
- Computes: `bad_frames / total_frames`

### 4. Decision Logic
- **≤10% bad frames** → `PASS` (artist-ready)
- **10-25% bad frames** → `PASS_WITH_CLEANUP`
- **>25% bad frames** → `FAIL` (fix before Maya)

## Usage

### Basic Usage
```bash
python pkl_motion_validator.py --pkl input.pkl --fps 30
```

### With JSON Report
```bash
python pkl_motion_validator.py --pkl input.pkl --fps 30 --output report.json
```

### With CSV Bad Frames
```bash
python pkl_motion_validator.py --pkl input.pkl --fps 30 --output report.json --csv bad_frames.csv
```

### Integrated in Pipeline
The validator is automatically run in `run_inference.sh` after PKL generation:
- Report saved to: `outputs/<name>/motion_validation.json`
- Bad frames CSV: `outputs/<name>/bad_frames.csv`

## Output

### Console Summary
```
======================================================================
PKL Motion Quality Validation Report
======================================================================
PKL File: outputs/smooth_fixed_v2_20260122_164820/smpl_output.pkl
FPS: 30.0
Total Frames: 120

Joint Angular Velocity:
  Max per joint: 1049.2°/sec
  Max global orientation: 2249.1°/sec
  Warnings: 560 joint-frame pairs
  Errors: 247 joint-frame pairs
  ...

Root Translation Jitter:
  Max displacement (magnitude): 347.260 cm/frame
  ...

Bad Frame Analysis:
  Bad frames: 116 / 120
  Bad frame ratio: 96.7%

======================================================================
❌ DECISION: FAIL (Fix Before Maya)
======================================================================
```

### JSON Report Structure
```json
{
  "pkl_path": "input.pkl",
  "fps": 30.0,
  "total_frames": 120,
  "angular_velocity": {
    "max_per_joint": [120.5, 95.3, ...],
    "max_global": 2249.1,
    "warning_count": 560,
    "error_count": 247
  },
  "root_jitter": {
    "max_displacement_cm_per_frame": 347.260,
    "max_per_axis_cm": [5.922, 4.183, 347.257],
    "warning_count": 113,
    "error_count": 95
  },
  "bad_frames": {
    "bad_frame_indices": [1, 2, 3, ...],
    "bad_frame_count": 116,
    "bad_frame_ratio": 0.967,
    "decision": "FAIL"
  },
  "thresholds": {
    "angular_velocity_warning_deg_per_sec": 120.0,
    "angular_velocity_error_deg_per_sec": 250.0,
    "root_jitter_warning_cm_per_frame": 0.3,
    "root_jitter_error_cm_per_frame": 1.0,
    "root_jitter_magnitude_warning_cm_per_frame": 5.0,
    "root_jitter_magnitude_error_cm_per_frame": 15.0,
    "bad_frame_ratio_pass": 0.10,
    "bad_frame_ratio_cleanup": 0.25
  }
}
```

### CSV Bad Frames
```csv
frame_index
1
2
3
...
```

## Exit Codes

- **0**: PASS or PASS_WITH_CLEANUP (motion is usable)
- **1**: FAIL (motion needs fixing)

Useful for CI/CD pipelines to block headless Maya animation if motion quality is poor.

## Dependencies

- NumPy (required)
- SciPy (optional, for better rotation distance computation)
- joblib or pickle (for PKL loading)

## Technical Details

### Angular Velocity Computation
- Uses geodesic distance on SO(3) for accurate rotation distance
- Falls back to simple magnitude difference if SciPy unavailable
- Handles axis-angle wrapping correctly

### Root Jitter Computation
- Checks both per-axis and magnitude displacement
- Z-axis often has large values in camera space, so magnitude thresholds are more lenient
- Per-axis checks are stricter for X/Y lateral movement

### Bad Frame Detection
- A frame is marked "bad" if:
  - Any joint exceeds angular velocity error threshold
  - Global orientation exceeds angular velocity error threshold
  - Root translation magnitude exceeds error threshold

## Integration

### In Python Code
```python
from pkl_motion_validator import validate_motion

results = validate_motion('input.pkl', fps=30.0)
decision = results['bad_frames']['decision']

if decision == 'FAIL':
    print("Motion quality too poor for Maya")
    # Block pipeline
else:
    print("Motion quality acceptable")
    # Continue to Maya
```

### In CI/CD
```bash
python pkl_motion_validator.py --pkl output.pkl --fps 30 || exit 1
# If validation fails, exit with error code
```

## Threshold Tuning

Thresholds are defined at the top of `pkl_motion_validator.py`:

```python
ANGULAR_VELOCITY_WARNING_DEG_PER_SEC = 120.0
ANGULAR_VELOCITY_ERROR_DEG_PER_SEC = 250.0
ROOT_JITTER_WARNING_CM_PER_FRAME = 0.3
ROOT_JITTER_ERROR_CM_PER_FRAME = 1.0
ROOT_JITTER_MAGNITUDE_WARNING_CM_PER_FRAME = 5.0
ROOT_JITTER_MAGNITUDE_ERROR_CM_PER_FRAME = 15.0
BAD_FRAME_RATIO_PASS = 0.10
BAD_FRAME_RATIO_CLEANUP = 0.25
```

Adjust these based on your specific requirements and motion characteristics.
