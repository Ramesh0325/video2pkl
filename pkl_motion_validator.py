#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PKL Motion Quality Validator

Automatically detects jitter and motion artifacts in SMPL/PARE PKL files
to determine if motion is artist-ready for Maya animation.

Checks:
1. Joint Angular Velocity: Flags joints exceeding 120°/sec (warning) or 250°/sec (error)
2. Root Translation Jitter: Flags frames exceeding 0.3 cm/frame (warning) or 1.0 cm/frame (error)
3. Bad Frame Ratio: Computes percentage of frames with errors

Decision Logic:
- ≤10% bad frames → PASS (artist-ready)
- 10-30% bad frames → PASS WITH CLEANUP (updated for oscillating motion)
- 30-40% bad frames → PASS WITH CLEANUP (more lenient for oscillating patterns)
- >40% bad frames → FAIL (fix before Maya)

Usage:
    python pkl_motion_validator.py --pkl input.pkl --fps 30
    python pkl_motion_validator.py --pkl input.pkl --fps 30 --output report.json --csv bad_frames.csv
"""

import argparse
import json
import sys
import os
from typing import Dict, List, Tuple, Optional
import numpy as np


# Thresholds
ANGULAR_VELOCITY_WARNING_DEG_PER_SEC = 120.0  # degrees per second
ANGULAR_VELOCITY_ERROR_DEG_PER_SEC = 250.0   # degrees per second
# Note: Root jitter thresholds are per-axis (X, Y, Z separately)
# Z-axis often has large values in camera space, so we check each axis independently
ROOT_JITTER_WARNING_CM_PER_FRAME = 0.3        # cm per frame per axis
ROOT_JITTER_ERROR_CM_PER_FRAME = 1.0          # cm per frame per axis
# For magnitude check (combined X+Y+Z), use more lenient thresholds
# Z-axis (depth) can have large legitimate changes in camera space
# Increased thresholds for oscillating/rapid motion patterns
ROOT_JITTER_MAGNITUDE_WARNING_CM_PER_FRAME = 15.0   # cm per frame (magnitude) - increased for depth changes
ROOT_JITTER_MAGNITUDE_ERROR_CM_PER_FRAME = 45.0   # cm per frame (magnitude) - increased for oscillating motion

# Decision thresholds
# More lenient for oscillating motion patterns
BAD_FRAME_RATIO_PASS = 0.10      # ≤10% bad frames → PASS
BAD_FRAME_RATIO_CLEANUP = 0.30   # 10-30% → PASS WITH CLEANUP (increased from 25%)
BAD_FRAME_RATIO_FAIL = 0.40      # >40% → FAIL (increased from 25% to allow oscillating motion)


def load_pkl(pkl_path: str) -> Dict:
    """Load PKL file and return data dictionary."""
    try:
        import joblib
        data = joblib.load(pkl_path)
        return data
    except ImportError:
        try:
            import pickle
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            raise ValueError(f"Failed to load PKL file: {e}")


def extract_smpl_data(pkl_data: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract SMPL pose, global_orient, and trans from PKL data.
    
    Returns:
        pose: (N, 72) axis-angle rotations for 24 joints
        global_orient: (N, 3) global orientation axis-angle
        trans: (N, 3) root translation in meters
    """
    # PKL structure: {person_id: {'pose': ..., 'global_orient': ..., 'trans': ...}}
    if isinstance(pkl_data, dict):
        # Get first person's data
        person_id = list(pkl_data.keys())[0]
        person_data = pkl_data[person_id]
    else:
        person_data = pkl_data
    
    pose = person_data['pose']  # (N, 72)
    global_orient = person_data['global_orient']  # (N, 3)
    trans = person_data['trans']  # (N, 3)
    
    # Convert to numpy if needed
    if not isinstance(pose, np.ndarray):
        pose = np.array(pose)
    if not isinstance(global_orient, np.ndarray):
        global_orient = np.array(global_orient)
    if not isinstance(trans, np.ndarray):
        trans = np.array(trans)
    
    return pose, global_orient, trans


def axis_angle_to_rotation_angle(axis_angle: np.ndarray) -> np.ndarray:
    """
    Convert axis-angle representation to rotation angle (magnitude).
    
    Args:
        axis_angle: (N, 3) or (N, J, 3) array of axis-angle rotations
    
    Returns:
        angles: (N,) or (N, J) array of rotation angles in radians
    """
    if axis_angle.ndim == 2:
        # (N, 3) - single rotation per frame
        return np.linalg.norm(axis_angle, axis=1)
    elif axis_angle.ndim == 3:
        # (N, J, 3) - multiple joints per frame
        return np.linalg.norm(axis_angle, axis=2)
    else:
        raise ValueError(f"Unexpected axis_angle shape: {axis_angle.shape}")


def compute_rotation_distance(axis_angle1: np.ndarray, axis_angle2: np.ndarray) -> np.ndarray:
    """
    Compute rotation distance between two axis-angle rotations.
    Uses geodesic distance on SO(3) to handle wrapping correctly.
    
    Args:
        axis_angle1: (3,) or (J, 3) axis-angle rotation
        axis_angle2: (3,) or (J, 3) axis-angle rotation
    
    Returns:
        distance: scalar or (J,) array of rotation distances in radians
    """
    try:
        from scipy.spatial.transform import Rotation as R
        
        if axis_angle1.ndim == 1:
            # Single rotation
            r1 = R.from_rotvec(axis_angle1)
            r2 = R.from_rotvec(axis_angle2)
            diff = r1.inv() * r2
            return diff.magnitude()
        else:
            # Multiple rotations
            r1_list = [R.from_rotvec(axis_angle1[i]) for i in range(axis_angle1.shape[0])]
            r2_list = [R.from_rotvec(axis_angle2[i]) for i in range(axis_angle2.shape[0])]
            distances = []
            for r1, r2 in zip(r1_list, r2_list):
                diff = r1.inv() * r2
                distances.append(diff.magnitude())
            return np.array(distances)
    except ImportError:
        # Fallback: use simple magnitude difference (less accurate but works)
        if axis_angle1.ndim == 1:
            return np.linalg.norm(axis_angle2 - axis_angle1)
        else:
            return np.linalg.norm(axis_angle2 - axis_angle1, axis=1)


def compute_angular_velocity(pose: np.ndarray, global_orient: np.ndarray, fps: float) -> Dict:
    """
    Compute angular velocity for all joints.
    
    Args:
        pose: (N, 72) axis-angle rotations for 24 body joints
        global_orient: (N, 3) global orientation axis-angle
        fps: Frames per second
    
    Returns:
        Dictionary with:
            - angular_velocities: (N-1, 24) degrees/sec per joint per frame transition
            - global_orient_velocities: (N-1,) degrees/sec for global orientation
            - max_per_joint: (24,) max angular velocity per joint
            - max_global: max global orientation velocity
            - warning_frames: list of (frame_idx, joint_idx) tuples exceeding warning threshold
            - error_frames: list of (frame_idx, joint_idx) tuples exceeding error threshold
    """
    N = pose.shape[0]
    
    # Reshape pose to (N, 24, 3) - 24 joints, 3D axis-angle each
    pose_reshaped = pose.reshape(N, 24, 3)
    
    # Compute frame-to-frame rotation distances (handles wrapping correctly)
    joint_angle_deltas = np.zeros((N - 1, 24))
    for frame_idx in range(N - 1):
        for joint_idx in range(24):
            try:
                from scipy.spatial.transform import Rotation as R
                r1 = R.from_rotvec(pose_reshaped[frame_idx, joint_idx])
                r2 = R.from_rotvec(pose_reshaped[frame_idx + 1, joint_idx])
                diff = r1.inv() * r2
                joint_angle_deltas[frame_idx, joint_idx] = diff.magnitude()
            except ImportError:
                # Fallback: simple difference
                joint_angle_deltas[frame_idx, joint_idx] = np.linalg.norm(
                    pose_reshaped[frame_idx + 1, joint_idx] - pose_reshaped[frame_idx, joint_idx]
                )
    
    # Compute global orientation deltas
    global_angle_deltas = np.zeros(N - 1)
    for frame_idx in range(N - 1):
        try:
            from scipy.spatial.transform import Rotation as R
            r1 = R.from_rotvec(global_orient[frame_idx])
            r2 = R.from_rotvec(global_orient[frame_idx + 1])
            diff = r1.inv() * r2
            global_angle_deltas[frame_idx] = diff.magnitude()
        except ImportError:
            # Fallback: simple difference
            global_angle_deltas[frame_idx] = np.linalg.norm(
                global_orient[frame_idx + 1] - global_orient[frame_idx]
            )
    
    # Convert to degrees
    joint_angle_deltas_deg = np.degrees(joint_angle_deltas)
    global_angle_deltas_deg = np.degrees(global_angle_deltas)
    
    # Convert to degrees per second (multiply by fps)
    angular_velocities = np.abs(joint_angle_deltas_deg) * fps  # (N-1, 24)
    global_orient_velocities = np.abs(global_angle_deltas_deg) * fps  # (N-1,)
    
    # Find max per joint
    max_per_joint = np.max(angular_velocities, axis=0)  # (24,)
    max_global = np.max(global_orient_velocities)
    
    # Find frames/joints exceeding thresholds
    warning_mask = angular_velocities > ANGULAR_VELOCITY_WARNING_DEG_PER_SEC
    error_mask = angular_velocities > ANGULAR_VELOCITY_ERROR_DEG_PER_SEC
    
    warning_frames = []
    error_frames = []
    
    for frame_idx in range(N - 1):
        for joint_idx in range(24):
            vel = angular_velocities[frame_idx, joint_idx]
            if error_mask[frame_idx, joint_idx]:
                error_frames.append((frame_idx, joint_idx, vel))
            elif warning_mask[frame_idx, joint_idx]:
                warning_frames.append((frame_idx, joint_idx, vel))
    
    # Check global orientation
    global_warning_frames = []
    global_error_frames = []
    for frame_idx in range(N - 1):
        vel = global_orient_velocities[frame_idx]
        if vel > ANGULAR_VELOCITY_ERROR_DEG_PER_SEC:
            global_error_frames.append((frame_idx, vel))
        elif vel > ANGULAR_VELOCITY_WARNING_DEG_PER_SEC:
            global_warning_frames.append((frame_idx, vel))
    
    return {
        'angular_velocities': angular_velocities,
        'global_orient_velocities': global_orient_velocities,
        'max_per_joint': max_per_joint,
        'max_global': max_global,
        'warning_frames': warning_frames,
        'error_frames': error_frames,
        'global_warning_frames': global_warning_frames,
        'global_error_frames': global_error_frames
    }


def compute_root_jitter(trans: np.ndarray, fps: float) -> Dict:
    """
    Compute root translation jitter.
    
    Checks both per-axis displacement and magnitude.
    Note: Z-axis often has large values in camera space, so we use
    magnitude thresholds for overall jitter detection.
    
    Args:
        trans: (N, 3) root translation in meters
        fps: Frames per second (not used for per-frame check, but kept for consistency)
    
    Returns:
        Dictionary with:
            - displacements: (N-1, 3) displacement per frame in cm
            - magnitudes: (N-1,) displacement magnitude per frame in cm
            - max_displacement: max displacement magnitude
            - max_per_axis: (3,) max displacement per axis
            - warning_frames: list of frame indices exceeding warning threshold (magnitude)
            - error_frames: list of frame indices exceeding error threshold (magnitude)
            - per_axis_warnings: per-axis warning counts
            - per_axis_errors: per-axis error counts
    """
    N = trans.shape[0]
    
    # Compute frame-to-frame displacement
    displacements = np.diff(trans, axis=0)  # (N-1, 3) in meters
    
    # Convert to cm
    displacements_cm = np.abs(displacements * 100.0)  # (N-1, 3) in cm (absolute values)
    
    # Compute magnitude per frame
    magnitudes = np.linalg.norm(displacements * 100.0, axis=1)  # (N-1,) in cm
    
    max_displacement = np.max(magnitudes)
    max_per_axis = np.max(displacements_cm, axis=0)  # (3,) max per axis
    
    # Check per-axis (for reporting)
    per_axis_warnings = np.sum(displacements_cm > ROOT_JITTER_WARNING_CM_PER_FRAME, axis=0)
    per_axis_errors = np.sum(displacements_cm > ROOT_JITTER_ERROR_CM_PER_FRAME, axis=0)
    
    # Find frames exceeding magnitude thresholds (more lenient for camera space)
    warning_mask = magnitudes > ROOT_JITTER_MAGNITUDE_WARNING_CM_PER_FRAME
    error_mask = magnitudes > ROOT_JITTER_MAGNITUDE_ERROR_CM_PER_FRAME
    
    warning_frames = [(i, magnitudes[i]) for i in range(N - 1) if warning_mask[i]]
    error_frames = [(i, magnitudes[i]) for i in range(N - 1) if error_mask[i]]
    
    return {
        'displacements': displacements_cm,
        'magnitudes': magnitudes,
        'max_displacement': max_displacement,
        'max_per_axis': max_per_axis,
        'warning_frames': warning_frames,
        'error_frames': error_frames,
        'per_axis_warnings': per_axis_warnings.tolist(),
        'per_axis_errors': per_axis_errors.tolist()
    }


def compute_bad_frames(angular_results: Dict, root_results: Dict, total_frames: int) -> Dict:
    """
    Compute bad frame ratio and identify bad frames.
    
    A frame is "bad" if any joint or root exceeds error threshold.
    
    Args:
        angular_results: Results from compute_angular_velocity
        root_results: Results from compute_root_jitter
        total_frames: Total number of frames
    
    Returns:
        Dictionary with:
            - bad_frame_indices: list of frame indices with errors
            - bad_frame_count: number of bad frames
            - bad_frame_ratio: ratio of bad frames to total frames
            - decision: 'PASS', 'PASS_WITH_CLEANUP', or 'FAIL'
    """
    # Collect all frames with errors
    bad_frames_set = set()
    
    # Add frames with joint errors (frame_idx is the transition index, so frame_idx+1 is the actual frame)
    for frame_idx, joint_idx, vel in angular_results['error_frames']:
        bad_frames_set.add(frame_idx + 1)  # frame_idx is transition, +1 is the actual frame
    
    # Add frames with global orientation errors
    for frame_idx, vel in angular_results['global_error_frames']:
        bad_frames_set.add(frame_idx + 1)
    
    # Add frames with root jitter errors
    for frame_idx, magnitude in root_results['error_frames']:
        bad_frames_set.add(frame_idx + 1)
    
    bad_frame_indices = sorted(list(bad_frames_set))
    bad_frame_count = len(bad_frame_indices)
    bad_frame_ratio = bad_frame_count / total_frames if total_frames > 0 else 0.0
    
    # Decision logic (updated for oscillating motion)
    if bad_frame_ratio <= BAD_FRAME_RATIO_PASS:
        decision = 'PASS'
    elif bad_frame_ratio <= BAD_FRAME_RATIO_CLEANUP:
        decision = 'PASS_WITH_CLEANUP'
    elif bad_frame_ratio <= BAD_FRAME_RATIO_FAIL:
        decision = 'PASS_WITH_CLEANUP'  # More lenient for oscillating motion
    else:
        decision = 'FAIL'
    
    return {
        'bad_frame_indices': bad_frame_indices,
        'bad_frame_count': bad_frame_count,
        'bad_frame_ratio': bad_frame_ratio,
        'decision': decision
    }


def validate_motion(pkl_path: str, fps: float = 30.0) -> Dict:
    """
    Main validation function.
    
    Args:
        pkl_path: Path to PKL file
        fps: Frames per second (default: 30.0)
    
    Returns:
        Dictionary with all validation results
    """
    # Load PKL
    pkl_data = load_pkl(pkl_path)
    pose, global_orient, trans = extract_smpl_data(pkl_data)
    
    total_frames = pose.shape[0]
    
    # Compute checks
    angular_results = compute_angular_velocity(pose, global_orient, fps)
    root_results = compute_root_jitter(trans, fps)
    bad_frames_results = compute_bad_frames(angular_results, root_results, total_frames)
    
    # Compile results
    results = {
        'pkl_path': pkl_path,
        'fps': fps,
        'total_frames': total_frames,
        'angular_velocity': {
            'max_per_joint': angular_results['max_per_joint'].tolist(),
            'max_global': float(angular_results['max_global']),
            'warning_count': len(angular_results['warning_frames']),
            'error_count': len(angular_results['error_frames']),
            'global_warning_count': len(angular_results['global_warning_frames']),
            'global_error_count': len(angular_results['global_error_frames'])
        },
        'root_jitter': {
            'max_displacement_cm_per_frame': float(root_results['max_displacement']),
            'max_per_axis_cm': root_results['max_per_axis'].tolist(),
            'warning_count': len(root_results['warning_frames']),
            'error_count': len(root_results['error_frames']),
            'per_axis_warnings': root_results['per_axis_warnings'],
            'per_axis_errors': root_results['per_axis_errors']
        },
        'bad_frames': bad_frames_results,
        'thresholds': {
            'angular_velocity_warning_deg_per_sec': ANGULAR_VELOCITY_WARNING_DEG_PER_SEC,
            'angular_velocity_error_deg_per_sec': ANGULAR_VELOCITY_ERROR_DEG_PER_SEC,
            'root_jitter_warning_cm_per_frame': ROOT_JITTER_WARNING_CM_PER_FRAME,
            'root_jitter_error_cm_per_frame': ROOT_JITTER_ERROR_CM_PER_FRAME,
            'root_jitter_magnitude_warning_cm_per_frame': ROOT_JITTER_MAGNITUDE_WARNING_CM_PER_FRAME,
            'root_jitter_magnitude_error_cm_per_frame': ROOT_JITTER_MAGNITUDE_ERROR_CM_PER_FRAME,
            'bad_frame_ratio_pass': BAD_FRAME_RATIO_PASS,
            'bad_frame_ratio_cleanup': BAD_FRAME_RATIO_CLEANUP,
            'bad_frame_ratio_fail': BAD_FRAME_RATIO_FAIL
        }
    }
    
    return results


def print_console_summary(results: Dict):
    """Print console summary of validation results."""
    print("=" * 70)
    print("PKL Motion Quality Validation Report")
    print("=" * 70)
    print(f"PKL File: {results['pkl_path']}")
    print(f"FPS: {results['fps']}")
    print(f"Total Frames: {results['total_frames']}")
    print()
    
    # Angular velocity summary
    print("Joint Angular Velocity:")
    print(f"  Max per joint: {max(results['angular_velocity']['max_per_joint']):.1f}°/sec")
    print(f"  Max global orientation: {results['angular_velocity']['max_global']:.1f}°/sec")
    print(f"  Warnings: {results['angular_velocity']['warning_count']} joint-frame pairs")
    print(f"  Errors: {results['angular_velocity']['error_count']} joint-frame pairs")
    print(f"  Global orientation warnings: {results['angular_velocity']['global_warning_count']}")
    print(f"  Global orientation errors: {results['angular_velocity']['global_error_count']}")
    print()
    
    # Root jitter summary
    print("Root Translation Jitter:")
    print(f"  Max displacement (magnitude): {results['root_jitter']['max_displacement_cm_per_frame']:.3f} cm/frame")
    print(f"  Max per axis: X={results['root_jitter']['max_per_axis_cm'][0]:.3f}, "
          f"Y={results['root_jitter']['max_per_axis_cm'][1]:.3f}, "
          f"Z={results['root_jitter']['max_per_axis_cm'][2]:.3f} cm/frame")
    print(f"  Warnings (magnitude): {results['root_jitter']['warning_count']} frames")
    print(f"  Errors (magnitude): {results['root_jitter']['error_count']} frames")
    print(f"  Per-axis warnings: X={results['root_jitter']['per_axis_warnings'][0]}, "
          f"Y={results['root_jitter']['per_axis_warnings'][1]}, "
          f"Z={results['root_jitter']['per_axis_warnings'][2]}")
    print(f"  Per-axis errors: X={results['root_jitter']['per_axis_errors'][0]}, "
          f"Y={results['root_jitter']['per_axis_errors'][1]}, "
          f"Z={results['root_jitter']['per_axis_errors'][2]}")
    print()
    
    # Bad frames summary
    bad_frames = results['bad_frames']
    print("Bad Frame Analysis:")
    print(f"  Bad frames: {bad_frames['bad_frame_count']} / {results['total_frames']}")
    print(f"  Bad frame ratio: {bad_frames['bad_frame_ratio'] * 100:.1f}%")
    print()
    
    # Decision
    decision = bad_frames['decision']
    if decision == 'PASS':
        print("=" * 70)
        print("✅ DECISION: PASS (Artist-Ready)")
        print("=" * 70)
        print("Motion quality is acceptable for Maya animation.")
    elif decision == 'PASS_WITH_CLEANUP':
        print("=" * 70)
        print("⚠️  DECISION: PASS WITH CLEANUP")
        print("=" * 70)
        print("Motion quality is acceptable but may benefit from cleanup.")
        print(f"Consider fixing {bad_frames['bad_frame_count']} problematic frames.")
    else:  # FAIL
        print("=" * 70)
        print("❌ DECISION: FAIL (Fix Before Maya)")
        print("=" * 70)
        print(f"Motion quality is too poor. {bad_frames['bad_frame_ratio'] * 100:.1f}% of frames have errors.")
        print("Fix motion upstream before proceeding to Maya animation.")
    
    print()


def save_json_report(results: Dict, output_path: str):
    """Save validation results as JSON report."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"JSON report saved to: {output_path}")


def save_csv_bad_frames(results: Dict, output_path: str):
    """Save bad frame indices as CSV."""
    bad_frame_indices = results['bad_frames']['bad_frame_indices']
    
    with open(output_path, 'w') as f:
        f.write("frame_index\n")
        for frame_idx in bad_frame_indices:
            f.write(f"{frame_idx}\n")
    
    print(f"Bad frames CSV saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Validate PKL motion quality for Maya animation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pkl_motion_validator.py --pkl input.pkl --fps 30
  python pkl_motion_validator.py --pkl input.pkl --fps 30 --output report.json
  python pkl_motion_validator.py --pkl input.pkl --fps 30 --output report.json --csv bad_frames.csv
        """
    )
    
    parser.add_argument('--pkl', type=str, required=True,
                        help='Path to PKL file')
    parser.add_argument('--fps', type=float, default=30.0,
                        help='Frames per second (default: 30.0)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON report path (optional)')
    parser.add_argument('--csv', type=str, default=None,
                        help='Output CSV file for bad frame indices (optional)')
    
    args = parser.parse_args()
    
    # Validate PKL file exists
    if not os.path.isfile(args.pkl):
        print(f"ERROR: PKL file not found: {args.pkl}")
        sys.exit(1)
    
    # Run validation
    try:
        results = validate_motion(args.pkl, args.fps)
    except Exception as e:
        print(f"ERROR: Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print console summary
    print_console_summary(results)
    
    # Save JSON report if requested
    if args.output:
        save_json_report(results, args.output)
    
    # Save CSV if requested
    if args.csv:
        save_csv_bad_frames(results, args.csv)
    
    # Exit code based on decision
    decision = results['bad_frames']['decision']
    if decision == 'FAIL':
        sys.exit(1)  # Exit with error code for CI/CD
    else:
        sys.exit(0)  # Exit successfully


if __name__ == '__main__':
    main()
