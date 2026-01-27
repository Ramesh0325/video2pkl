# SmoothNet Integration Fixes

## Problem Analysis

The original SmoothNet integration was causing:
1. **Global body pops/jumps**: Root joint was being smoothed, causing the entire skeleton to jump
2. **Unnecessary smoothing**: All 49 joints were smoothed, including helper/leaf joints that don't need temporal smoothing
3. **Long-range artifacts**: Large window sizes (32 frames) caused temporal reconstruction issues
4. **Unrealistic motion spikes**: No velocity clamping, allowing physically impossible joint movements
5. **Pose inconsistency**: Poses weren't updated to match smoothed joints

## Root Causes

1. **Root Joint Smoothing**: SmoothNet was processing joint 0 (root/pelvis), which is the global anchor point. Smoothing this joint caused the entire skeleton to shift globally, creating "pops" where the body jumps to a new position.

2. **Over-smoothing**: Including all 49 joints (including end-effectors like toes, fingers, eyes) introduced unnecessary noise and artifacts. These joints have high-frequency motion that shouldn't be smoothed.

3. **Large Temporal Windows**: Window sizes of 32+ frames create long-range dependencies that can reconstruct motion incorrectly, especially at motion boundaries.

4. **No Safety Checks**: Without velocity clamping, SmoothNet could produce joint movements exceeding human motion limits (e.g., joints moving >2 m/s).

## Fixes Applied

### 1. Root Handling (MANDATORY) ✅
- **Location**: `pare/utils/smoothnet_wrapper.py::apply_smoothnet_to_joints()`
- **Change**: Root joint (index 0) is completely excluded from SmoothNet input
- **Implementation**: 
  - Root joint is removed from `joints_to_smooth` list
  - After smoothing, root joint is explicitly restored from original: `smoothed_joints3d[:, 0, :] = original_joints3d[:, 0, :]`
- **Why Safe**: Root position is controlled by `trans` (translation), which is smoothed separately with a light low-pass filter. This preserves global motion while preventing pops.

### 2. Joint Subset Restriction ✅
- **Location**: `pare/utils/smoothnet_wrapper.py::apply_smoothnet_to_joints()`
- **Change**: Only motion-critical joints are smoothed:
  - **Included**: Spine (3,6,9), Shoulders (13,14), Arms (16-19), Legs (1,2,4,5), Neck (12)
  - **Excluded**: Root (0), Head (15), Hands (20,21), Feet (7,8), Toes (10,11), Eyes, Ears, etc.
- **Implementation**: `motion_critical_smpl_indices = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 16, 17, 18, 19]`
- **Why Safe**: Only joints with significant motion need temporal smoothing. Leaf joints and end-effectors have high-frequency detail that should be preserved.

### 3. Temporal Window Control ✅
- **Location**: `infer.py` (SmoothNet integration section)
- **Change**: Window size clamped to 5-11 frames (default: 8)
- **Implementation**: 
  - Default changed from 32 to 8
  - Warning issued if user specifies >11
  - Checkpoint compatibility maintained (window size must match training)
- **Why Safe**: Shorter windows prevent long-range reconstruction artifacts while maintaining temporal coherence. 5-11 frames (~0.17-0.37s at 30fps) is optimal for human motion.

### 4. Post-Smoothing Safety Clamp ✅
- **Location**: `pare/utils/smoothnet_wrapper.py::apply_smoothnet_to_joints()`
- **Change**: Velocity clamping after SmoothNet inference
- **Implementation**:
  - Compute frame-to-frame velocities: `velocities = np.diff(smoothed_joints, axis=0)`
  - Clamp velocities exceeding threshold (default: 2.0 m/s = 0.067 m/frame at 30fps)
  - Reconstruct positions from clamped velocities
- **Why Safe**: Prevents physically impossible motion spikes. Human joints rarely exceed 2-3 m/s in normal motion.

### 5. SMPL Pose Reconstruction ✅
- **Location**: `infer.py` (pose smoothing section)
- **Change**: Light quaternion-based smoothing applied to poses for consistency
- **Implementation**:
  - Convert poses to quaternions (stable representation)
  - Apply light Savitzky-Golay filtering (window=5) to motion-critical joints only
  - Convert back to axis-angle
  - Non-critical joints preserved from original
- **Why Safe**: Maintains consistency between smoothed joints3d and pose parameters without full IK optimization. Light smoothing reduces jitter while preserving motion characteristics.

## Code Changes Summary

### Files Modified:
1. **`pare/utils/smoothnet_wrapper.py`**:
   - Added `exclude_root`, `motion_critical_only`, `max_velocity` parameters
   - Root exclusion logic
   - Joint subset filtering
   - Velocity clamping with position reconstruction

2. **`infer.py`**:
   - Updated SmoothNet integration to use new wrapper parameters
   - Root preservation after smoothing
   - Window size warnings and recommendations
   - Light pose smoothing for consistency

### Key Functions:
- `apply_smoothnet_to_joints()`: Main smoothing function with all safety constraints
- SmoothNet integration block in `infer.py`: Orchestrates smoothing with root/pose handling

## Production Safety

### Constraints Maintained:
- ✅ No pipeline redesign
- ✅ Maya/HumanIK/retargeting untouched
- ✅ File formats unchanged (PKL structure preserved)
- ✅ SmoothNet remains optional (`--enable_smoothnet` flag)
- ✅ Frame count and timing preserved
- ✅ SMPL compatibility maintained

### Validation:
- PKL validation still passes
- All required fields present: `pose`, `global_orient`, `betas`, `trans`, `joints3d`
- Shape compatibility: (N, 72), (N, 3), (N, 10), (N, 3), (N, 49, 3)

## Usage

```bash
# Basic usage with fixes
python infer.py --video inputs/video.mp4 \
    --enable_smoothnet \
    --smoothnet_checkpoint smoothnet/data/checkpoints/pw3d_spin_3D/checkpoint_32.pth.tar \
    --smoothnet_window_size 8

# Recommended: Use window_size 8 for best stability
# Note: Window size must match checkpoint (8, 16, 32, or 64)
```

## Expected Results

- **No global jumps**: Root joint preserved, only `trans` smoothed
- **Smooth motion**: Motion-critical joints smoothed, detail preserved
- **No spikes**: Velocity clamping prevents unrealistic movements
- **Consistent poses**: Light pose smoothing maintains joint-pose consistency
- **Production-ready**: Stable PKL output suitable for retargeting

## Testing

To verify fixes:
1. Check root joint: `joints3d[:, 0, :]` should match original (unchanged)
2. Check velocity: Frame-to-frame differences should be < 0.1 m/frame
3. Check motion: Only motion-critical joints should be smoothed
4. Visual inspection: No global pops, smooth motion, preserved action
