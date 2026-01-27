#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Motion Stabilizer for AI-Generated PKL Motion Data

Eliminates single-frame jitter and Euler rotation spikes before Maya retargeting.
Designed for monocular pose estimation (PARE/SMPL) output.

Pipeline Position:
    Pose Estimation → SmoothNet → PKL → THIS STABILIZATION → Headless Maya → Retargeting

Usage:
    python motion_stabilizer.py --input input.pkl --output clean.pkl --fps 30
    python motion_stabilizer.py --input input.pkl --output clean.pkl --fps 30 --window 5 --max_velocity 200
"""

import argparse
import json
import sys
import os
from typing import Dict, List, Tuple, Set
import numpy as np


# Default thresholds
DEFAULT_FPS = 30.0
DEFAULT_OUTLIER_THRESHOLD_DEG = 200.0  # Flag frames with >200° rotation in one frame
DEFAULT_MAX_ANGULAR_VELOCITY_DEG_PER_SEC = 210.0  # Clamp to 210°/sec
DEFAULT_SMOOTHING_WINDOW = 5  # Temporal smoothing window (frames)
MAX_OUTLIER_RATIO = 0.30  # If >30% frames are outliers, mark as not artist-ready


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


def save_pkl(data: Dict, output_path: str):
    """Save PKL file."""
    try:
        import joblib
        joblib.dump(data, output_path)
    except ImportError:
        import pickle
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)


def extract_smpl_data(pkl_data: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Extract SMPL pose, global_orient, trans, and metadata from PKL data.
    
    Returns:
        pose: (N, 72) axis-angle rotations for 24 joints
        global_orient: (N, 3) global orientation axis-angle
        trans: (N, 3) root translation
        metadata: Dictionary with other data (betas, frame_ids, etc.)
    """
    if isinstance(pkl_data, dict):
        # Get first person's data
        person_id = list(pkl_data.keys())[0]
        person_data = pkl_data[person_id]
    else:
        person_data = pkl_data
    
    pose = np.array(person_data['pose'])  # (N, 72)
    global_orient = np.array(person_data['global_orient'])  # (N, 3)
    trans = np.array(person_data['trans'])  # (N, 3)
    
    # Extract metadata
    metadata = {}
    for key in person_data.keys():
        if key not in ['pose', 'global_orient', 'trans']:
            metadata[key] = person_data[key]
    
    return pose, global_orient, trans, metadata


def axis_angle_to_quaternion(axis_angle: np.ndarray) -> np.ndarray:
    """
    Convert axis-angle to quaternion with normalization.
    
    Args:
        axis_angle: (N, 3) or (N, J, 3) array of axis-angle rotations
    
    Returns:
        quaternions: (N, 4) or (N, J, 4) array of quaternions [w, x, y, z]
    """
    try:
        from scipy.spatial.transform import Rotation as R
        
        if axis_angle.ndim == 2:
            # (N, 3) - single rotation per frame
            rots = [R.from_rotvec(axis_angle[i]) for i in range(len(axis_angle))]
            quats = np.array([r.as_quat() for r in rots])  # (N, 4) [w, x, y, z]
        elif axis_angle.ndim == 3:
            # (N, J, 3) - multiple joints per frame
            N, J, _ = axis_angle.shape
            quats = np.zeros((N, J, 4))
            for i in range(N):
                for j in range(J):
                    r = R.from_rotvec(axis_angle[i, j])
                    quats[i, j] = r.as_quat()
        else:
            raise ValueError(f"Unexpected axis_angle shape: {axis_angle.shape}")
        
        return quats
    except ImportError:
        # Fallback: manual conversion
        if axis_angle.ndim == 2:
            angle = np.linalg.norm(axis_angle, axis=1, keepdims=True)
            axis = axis_angle / (angle + 1e-8)
            half_angle = angle / 2.0
            w = np.cos(half_angle)
            xyz = axis * np.sin(half_angle)
            quats = np.concatenate([w, xyz], axis=1)
        else:
            # (N, J, 3)
            N, J, _ = axis_angle.shape
            quats = np.zeros((N, J, 4))
            for i in range(N):
                for j in range(J):
                    angle = np.linalg.norm(axis_angle[i, j])
                    if angle < 1e-8:
                        quats[i, j] = np.array([1.0, 0.0, 0.0, 0.0])
                    else:
                        axis = axis_angle[i, j] / angle
                        half_angle = angle / 2.0
                        quats[i, j, 0] = np.cos(half_angle)
                        quats[i, j, 1:] = axis * np.sin(half_angle)
        return quats


def enforce_quaternion_continuity(quats: np.ndarray) -> np.ndarray:
    """
    Enforce shortest-arc continuity between consecutive frames.
    Flip quaternion sign if dot product < 0.
    
    Args:
        quats: (N, 4) or (N, J, 4) array of quaternions
    
    Returns:
        continuous_quats: Same shape, with continuity enforced
    """
    continuous_quats = quats.copy()
    
    if quats.ndim == 2:
        # (N, 4) - single rotation per frame
        for i in range(1, len(continuous_quats)):
            dot = np.dot(continuous_quats[i-1], continuous_quats[i])
            if dot < 0:
                continuous_quats[i] = -continuous_quats[i]
    elif quats.ndim == 3:
        # (N, J, 4) - multiple joints per frame
        for i in range(1, len(continuous_quats)):
            for j in range(continuous_quats.shape[1]):
                dot = np.dot(continuous_quats[i-1, j], continuous_quats[i, j])
                if dot < 0:
                    continuous_quats[i, j] = -continuous_quats[i, j]
    
    return continuous_quats


def detect_outlier_frames(quats: np.ndarray, threshold_deg: float, fps: float) -> Set[int]:
    """
    Detect single-frame outliers where rotation > threshold in one frame.
    
    Args:
        quats: (N, J, 4) quaternions for all joints
        threshold_deg: Threshold in degrees (e.g., 200°)
        fps: Frames per second
    
    Returns:
        outlier_frames: Set of frame indices that are outliers
    """
    try:
        from scipy.spatial.transform import Rotation as R
        
        N, J, _ = quats.shape
        outlier_frames = set()
        threshold_rad = np.radians(threshold_deg)
        
        for joint_idx in range(J):
            for frame_idx in range(1, N):
                # Compute rotation distance between consecutive frames
                r_prev = R.from_quat(quats[frame_idx - 1, joint_idx])
                r_curr = R.from_quat(quats[frame_idx, joint_idx])
                rot_diff = r_prev.inv() * r_curr
                angle_diff = rot_diff.magnitude()
                
                # Convert to degrees per second
                angle_diff_deg_per_sec = np.degrees(angle_diff) * fps
                
                # Check if this is an outlier
                if angle_diff_deg_per_sec > threshold_deg:
                    outlier_frames.add(frame_idx)
        
        return outlier_frames
    except ImportError:
        # Fallback: simpler check using quaternion distance
        N, J, _ = quats.shape
        outlier_frames = set()
        threshold_rad = np.radians(threshold_deg / fps)  # Per-frame threshold
        
        for joint_idx in range(J):
            for frame_idx in range(1, N):
                # Quaternion distance: 1 - |dot(q1, q2)|
                dot = np.abs(np.dot(quats[frame_idx - 1, joint_idx], quats[frame_idx, joint_idx]))
                angle = 2 * np.arccos(np.clip(dot, -1, 1))
                
                if angle > threshold_rad:
                    outlier_frames.add(frame_idx)
        
        return outlier_frames


def slerp_quaternion(q1: np.ndarray, q2: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation (SLERP) between two quaternions.
    
    Args:
        q1: (4,) quaternion
        q2: (4,) quaternion
        t: Interpolation parameter [0, 1]
    
    Returns:
        q_interp: (4,) interpolated quaternion
    """
    # Normalize
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # Ensure shortest path
    dot = np.dot(q1, q2)
    if dot < 0:
        q2 = -q2
        dot = -dot
    
    # Clamp dot product to valid range
    dot = np.clip(dot, -1.0, 1.0)
    
    # If quaternions are very close, use linear interpolation
    if abs(dot) > 0.9995:
        q_interp = (1 - t) * q1 + t * q2
        return q_interp / np.linalg.norm(q_interp)
    
    # SLERP
    theta = np.arccos(abs(dot))
    sin_theta = np.sin(theta)
    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta
    
    q_interp = w1 * q1 + w2 * q2
    return q_interp / np.linalg.norm(q_interp)


def replace_outlier_frames(quats: np.ndarray, outlier_frames: Set[int]) -> Tuple[np.ndarray, int]:
    """
    Replace outlier frames using SLERP interpolation between prev and next valid frames.
    Handles cases with many consecutive outliers by using weighted interpolation.
    
    Args:
        quats: (N, J, 4) quaternions
        outlier_frames: Set of frame indices to replace
    
    Returns:
        fixed_quats: (N, J, 4) quaternions with outliers replaced
        frames_fixed: Number of frames fixed
    """
    fixed_quats = quats.copy()
    frames_fixed = 0
    
    N, J, _ = quats.shape
    
    # Sort outlier frames
    sorted_outliers = sorted(outlier_frames)
    
    for frame_idx in sorted_outliers:
        # Find previous valid frame
        prev_idx = frame_idx - 1
        while prev_idx >= 0 and prev_idx in outlier_frames:
            prev_idx -= 1
        
        # Find next valid frame
        next_idx = frame_idx + 1
        while next_idx < N and next_idx in outlier_frames:
            next_idx += 1
        
        # Replace using SLERP
        if prev_idx >= 0 and next_idx < N:
            # Calculate interpolation parameter based on distance
            total_distance = next_idx - prev_idx
            local_distance = frame_idx - prev_idx
            t = local_distance / total_distance if total_distance > 0 else 0.5
            
            # Interpolate between prev and next
            for joint_idx in range(J):
                q_prev = fixed_quats[prev_idx, joint_idx]  # Use fixed version if available
                q_next = quats[next_idx, joint_idx]  # Use original for next
                fixed_quats[frame_idx, joint_idx] = slerp_quaternion(q_prev, q_next, t)
            frames_fixed += 1
        elif prev_idx >= 0:
            # Only prev available, copy it (or use smoothed version)
            fixed_quats[frame_idx] = fixed_quats[prev_idx]
            frames_fixed += 1
        elif next_idx < N:
            # Only next available, copy it
            fixed_quats[frame_idx] = quats[next_idx]
            frames_fixed += 1
        else:
            # No valid neighbors - keep original (shouldn't happen)
            pass
    
    return fixed_quats, frames_fixed


def smooth_quaternions_temporal(quats: np.ndarray, window_size: int, exclude_frames: Set[int]) -> np.ndarray:
    """
    Apply light temporal smoothing in quaternion space.
    Exclude already-fixed frames from smoothing.
    
    Args:
        quats: (N, J, 4) quaternions
        window_size: Smoothing window size (3-5 frames recommended)
        exclude_frames: Frames to exclude from smoothing (already fixed)
    
    Returns:
        smoothed_quats: (N, J, 4) smoothed quaternions
    """
    try:
        from scipy import signal
        
        N, J, _ = quats.shape
        smoothed_quats = quats.copy()
        
        # Smooth each quaternion component independently
        for joint_idx in range(J):
            for dim in range(4):
                quat_component = quats[:, joint_idx, dim]
                
                # Create mask for frames to smooth
                smooth_mask = np.ones(N, dtype=bool)
                for frame_idx in exclude_frames:
                    smooth_mask[frame_idx] = False
                
                if np.sum(smooth_mask) > window_size:
                    # Apply Savitzky-Golay filter only to non-excluded frames
                    # For excluded frames, keep original
                    valid_indices = np.where(smooth_mask)[0]
                    
                    if len(valid_indices) > window_size:
                        # Extract valid segment
                        valid_component = quat_component[valid_indices]
                        
                        # Smooth
                        window_length = min(window_size, len(valid_component))
                        if window_length >= 3 and window_length % 2 == 1:
                            smoothed_valid = signal.savgol_filter(
                                valid_component,
                                window_length=window_length,
                                polyorder=min(2, window_length - 1)
                            )
                            # Write back
                            smoothed_quats[valid_indices, joint_idx, dim] = smoothed_valid
        
        return smoothed_quats
    except ImportError:
        # Fallback: simple moving average
        N, J, _ = quats.shape
        smoothed_quats = quats.copy()
        
        kernel_size = min(window_size, N)
        kernel = np.ones(kernel_size) / kernel_size
        
        for joint_idx in range(J):
            for dim in range(4):
                quat_component = quats[:, joint_idx, dim]
                smoothed_component = np.convolve(quat_component, kernel, mode='same')
                smoothed_quats[:, joint_idx, dim] = smoothed_component
        
        return smoothed_quats


def detect_root_outliers(trans: np.ndarray, threshold_cm_per_frame: float = 50.0) -> Set[int]:
    """
    Detect outlier frames in root translation based on frame-to-frame displacement.
    
    Args:
        trans: (N, 3) root translation in meters
        threshold_cm_per_frame: Maximum allowed displacement in cm/frame
    
    Returns:
        Set of frame indices (0-indexed, the frame AFTER the jump) that are outliers
    """
    N = trans.shape[0]
    outlier_frames = set()
    
    if N < 2:
        return outlier_frames
    
    # Compute frame-to-frame displacement
    displacements = np.diff(trans, axis=0)  # (N-1, 3) in meters
    magnitudes_cm = np.linalg.norm(displacements * 100.0, axis=1)  # (N-1,) in cm
    
    # Flag frames where displacement exceeds threshold
    for i in range(N - 1):
        if magnitudes_cm[i] > threshold_cm_per_frame:
            # Mark the frame AFTER the jump as an outlier
            outlier_frames.add(i + 1)
    
    return outlier_frames


def smooth_root_translation(trans: np.ndarray, window_size: int = 5, outlier_frames: Set[int] = None) -> np.ndarray:
    """
    Smooth root translation with adaptive filtering.
    More aggressive smoothing for outlier frames and Z-axis (depth).
    Prevents vertical (Y) jitter while preserving locomotion.
    
    Args:
        trans: (N, 3) root translation
        window_size: Smoothing window size (default: 5, increased for better smoothing)
        outlier_frames: Set of frame indices to apply more aggressive smoothing
    
    Returns:
        smoothed_trans: (N, 3) smoothed translation
    """
    if outlier_frames is None:
        outlier_frames = set()
    
    try:
        from scipy import signal
        
        smoothed_trans = trans.copy()
        N = len(trans)
        
        # Different strategies for different axes
        # X/Y: moderate smoothing (preserve horizontal motion)
        # Z (depth): aggressive smoothing (camera-space depth can have large but smooth changes)
        for dim in range(3):
            if N > window_size:
                # Z-axis (depth) needs more aggressive smoothing for oscillating motion
                if dim == 2:  # Z-axis
                    # Use larger window for Z-axis (even more aggressive)
                    z_window = min(window_size + 6, N)
                    if z_window >= 3:
                        if z_window % 2 == 0:
                            z_window -= 1
                        
                        # Apply aggressive smoothing to Z-axis (first pass)
                        smoothed_trans[:, dim] = signal.savgol_filter(
                            trans[:, dim],
                            window_length=z_window,
                            polyorder=min(2, z_window - 1)
                        )
                        
                        # Apply second pass for refinement (always for Z-axis)
                        z_window2 = max(5, z_window - 4)
                        if z_window2 % 2 == 0:
                            z_window2 -= 1
                        if z_window2 >= 5:
                            smoothed_trans[:, dim] = signal.savgol_filter(
                                smoothed_trans[:, dim],
                                window_length=z_window2,
                                polyorder=min(2, z_window2 - 1)
                            )
                        
                        # Third pass for very oscillating motion (light smoothing)
                        if len(outlier_frames) > 10 or N > 50:
                            z_window3 = max(5, z_window2 - 2)
                            if z_window3 % 2 == 0:
                                z_window3 -= 1
                            if z_window3 >= 5:
                                smoothed_trans[:, dim] = signal.savgol_filter(
                                    smoothed_trans[:, dim],
                                    window_length=z_window3,
                                    polyorder=min(1, z_window3 - 1)
                                )
                else:  # X and Y axes
                    # Use standard window for X/Y
                    window_length = min(window_size, N)
                    if window_length >= 3:
                        if window_length % 2 == 0:
                            window_length -= 1
                        
                        # Apply standard smoothing
                        smoothed_trans[:, dim] = signal.savgol_filter(
                            trans[:, dim],
                            window_length=window_length,
                            polyorder=min(2, window_length - 1)
                        )
                
                # Apply additional smoothing to outlier frames (all axes)
                if len(outlier_frames) > 0:
                    # Use larger window for outliers
                    outlier_window = min(window_size + 4, N)
                    if outlier_window % 2 == 0:
                        outlier_window -= 1
                    
                    for frame_idx in outlier_frames:
                        if 0 <= frame_idx < N:
                            # Extract local window around outlier
                            start = max(0, frame_idx - outlier_window // 2)
                            end = min(N, frame_idx + outlier_window // 2 + 1)
                            
                            if end - start >= 3:
                                # Smooth this local region more aggressively
                                local_window = min(outlier_window, end - start)
                                if local_window % 2 == 0:
                                    local_window -= 1
                                
                                if local_window >= 3:
                                    local_smoothed = signal.savgol_filter(
                                        smoothed_trans[start:end, dim],
                                        window_length=local_window,
                                        polyorder=min(1, local_window - 1)
                                    )
                                    smoothed_trans[start:end, dim] = local_smoothed
        
        return smoothed_trans
    except ImportError:
        # Fallback: simple moving average with larger kernel
        kernel_size = min(window_size * 2, len(trans))
        kernel = np.ones(kernel_size) / kernel_size
        
        smoothed_trans = trans.copy()
        for dim in range(3):
            smoothed_trans[:, dim] = np.convolve(trans[:, dim], kernel, mode='same')
        
        return smoothed_trans


def clamp_angular_velocity(quats: np.ndarray, max_velocity_deg_per_sec: float, fps: float) -> Tuple[np.ndarray, int]:
    """
    Clamp angular velocity to max_velocity_deg_per_sec.
    Damp only the excess rotation.
    
    Args:
        quats: (N, J, 4) quaternions
        max_velocity_deg_per_sec: Maximum allowed angular velocity
        fps: Frames per second
    
    Returns:
        clamped_quats: (N, J, 4) quaternions with velocity clamped
        joints_clamped: Number of joint-frame pairs clamped
    """
    try:
        from scipy.spatial.transform import Rotation as R
        
        N, J, _ = quats.shape
        clamped_quats = quats.copy()
        joints_clamped = 0
        max_velocity_rad_per_frame = np.radians(max_velocity_deg_per_sec / fps)
        
        for joint_idx in range(J):
            for frame_idx in range(1, N):
                r_prev = R.from_quat(clamped_quats[frame_idx - 1, joint_idx])
                r_curr = R.from_quat(clamped_quats[frame_idx, joint_idx])
                rot_diff = r_prev.inv() * r_curr
                angle_diff = rot_diff.magnitude()
                
                if angle_diff > max_velocity_rad_per_frame:
                    # Clamp: scale down the rotation
                    scale = max_velocity_rad_per_frame / angle_diff
                    rot_vec = rot_diff.as_rotvec()
                    rot_vec_clamped = rot_vec * scale
                    rot_diff_clamped = R.from_rotvec(rot_vec_clamped)
                    r_clamped = r_prev * rot_diff_clamped
                    clamped_quats[frame_idx, joint_idx] = r_clamped.as_quat()
                    joints_clamped += 1
        
        return clamped_quats, joints_clamped
    except ImportError:
        # Fallback: simpler approach
        N, J, _ = quats.shape
        clamped_quats = quats.copy()
        joints_clamped = 0
        
        # Use quaternion distance as proxy
        max_velocity_rad_per_frame = np.radians(max_velocity_deg_per_sec / fps)
        
        for joint_idx in range(J):
            for frame_idx in range(1, N):
                dot = np.abs(np.dot(clamped_quats[frame_idx - 1, joint_idx], 
                                   clamped_quats[frame_idx, joint_idx]))
                angle = 2 * np.arccos(np.clip(dot, -1, 1))
                
                if angle > max_velocity_rad_per_frame:
                    # Interpolate towards previous frame
                    t = max_velocity_rad_per_frame / angle
                    clamped_quats[frame_idx, joint_idx] = slerp_quaternion(
                        clamped_quats[frame_idx - 1, joint_idx],
                        clamped_quats[frame_idx, joint_idx],
                        t
                    )
                    joints_clamped += 1
        
        return clamped_quats, joints_clamped


def quaternion_to_axis_angle(quats: np.ndarray) -> np.ndarray:
    """
    Convert quaternions back to axis-angle format.
    
    Args:
        quats: (N, 4) or (N, J, 4) quaternions
    
    Returns:
        axis_angle: (N, 3) or (N, J, 3) axis-angle rotations
    """
    try:
        from scipy.spatial.transform import Rotation as R
        
        if quats.ndim == 2:
            # (N, 4)
            rots = [R.from_quat(quats[i]) for i in range(len(quats))]
            axis_angle = np.array([r.as_rotvec() for r in rots])
        elif quats.ndim == 3:
            # (N, J, 4)
            N, J, _ = quats.shape
            axis_angle = np.zeros((N, J, 3))
            for i in range(N):
                for j in range(J):
                    r = R.from_quat(quats[i, j])
                    axis_angle[i, j] = r.as_rotvec()
        else:
            raise ValueError(f"Unexpected quats shape: {quats.shape}")
        
        return axis_angle
    except ImportError:
        # Fallback: manual conversion
        if quats.ndim == 2:
            # (N, 4)
            w = quats[:, 0]
            xyz = quats[:, 1:]
            angle = 2 * np.arccos(np.clip(w, -1, 1))
            sin_half = np.sin(angle / 2.0)
            axis = xyz / (sin_half[:, np.newaxis] + 1e-8)
            axis_angle = axis * angle[:, np.newaxis]
        else:
            # (N, J, 4)
            N, J, _ = quats.shape
            axis_angle = np.zeros((N, J, 3))
            for i in range(N):
                for j in range(J):
                    w = quats[i, j, 0]
                    xyz = quats[i, j, 1:]
                    angle = 2 * np.arccos(np.clip(w, -1, 1))
                    if angle < 1e-8:
                        axis_angle[i, j] = np.zeros(3)
                    else:
                        sin_half = np.sin(angle / 2.0)
                        axis = xyz / (sin_half + 1e-8)
                        axis_angle[i, j] = axis * angle
        return axis_angle


def compute_max_angular_velocity(quats: np.ndarray, fps: float) -> float:
    """Compute maximum angular velocity in deg/sec."""
    try:
        from scipy.spatial.transform import Rotation as R
        
        N, J, _ = quats.shape
        max_vel = 0.0
        
        for joint_idx in range(J):
            for frame_idx in range(1, N):
                r_prev = R.from_quat(quats[frame_idx - 1, joint_idx])
                r_curr = R.from_quat(quats[frame_idx, joint_idx])
                rot_diff = r_prev.inv() * r_curr
                angle_diff = rot_diff.magnitude()
                vel_deg_per_sec = np.degrees(angle_diff) * fps
                max_vel = max(max_vel, vel_deg_per_sec)
        
        return max_vel
    except ImportError:
        # Fallback
        N, J, _ = quats.shape
        max_vel = 0.0
        
        for joint_idx in range(J):
            for frame_idx in range(1, N):
                dot = np.abs(np.dot(quats[frame_idx - 1, joint_idx], 
                                   quats[frame_idx, joint_idx]))
                angle = 2 * np.arccos(np.clip(dot, -1, 1))
                vel_deg_per_sec = np.degrees(angle) * fps
                max_vel = max(max_vel, vel_deg_per_sec)
        
        return max_vel


def stabilize_motion(input_pkl: str, output_pkl: str, fps: float = DEFAULT_FPS,
                     outlier_threshold: float = DEFAULT_OUTLIER_THRESHOLD_DEG,
                     max_velocity: float = DEFAULT_MAX_ANGULAR_VELOCITY_DEG_PER_SEC,
                     window_size: int = DEFAULT_SMOOTHING_WINDOW) -> Dict:
    """
    Main stabilization function.
    
    Returns:
        Dictionary with stabilization report
    """
    # Load PKL
    pkl_data = load_pkl(input_pkl)
    pose, global_orient, trans, metadata = extract_smpl_data(pkl_data)
    
    N = pose.shape[0]
    pose_reshaped = pose.reshape(N, 24, 3)  # (N, 24, 3)
    
    # Combine global_orient and body pose for processing
    full_pose = np.concatenate([global_orient[:, np.newaxis, :], pose_reshaped], axis=1)  # (N, 25, 3)
    
    # Compute max angular velocity before
    full_pose_quats_before = axis_angle_to_quaternion(full_pose)  # (N, 25, 4)
    full_pose_quats_before = enforce_quaternion_continuity(full_pose_quats_before)
    max_vel_before = compute_max_angular_velocity(full_pose_quats_before, fps)
    
    # Step 2: Convert to quaternions with continuity
    full_pose_quats = axis_angle_to_quaternion(full_pose)
    full_pose_quats = enforce_quaternion_continuity(full_pose_quats)
    
    # Step 3: Detect outliers (for body joints only, exclude global_orient from outlier detection)
    body_pose_quats = full_pose_quats[:, 1:, :]  # (N, 24, 4) - exclude global_orient
    outlier_frames = detect_outlier_frames(body_pose_quats, outlier_threshold, fps)
    
    # Also check global_orient separately
    global_orient_quats = full_pose_quats[:, 0:1, :]  # (N, 1, 4)
    global_outlier_frames = detect_outlier_frames(global_orient_quats, outlier_threshold, fps)
    outlier_frames.update(global_outlier_frames)
    
    outlier_ratio = len(outlier_frames) / N if N > 0 else 0.0
    
    # Check if too many outliers - but still try to fix what we can
    if outlier_ratio > MAX_OUTLIER_RATIO:
        print(f"WARNING: High outlier ratio ({outlier_ratio*100:.1f}% > {MAX_OUTLIER_RATIO*100}%)")
        print(f"  This motion has many problematic frames, but will attempt to fix them.")
        print(f"  Consider adjusting outlier_threshold or fixing motion upstream.")
        # Don't fail - try to fix what we can
        # For very high ratios, we'll still process but mark as needing attention
    
    # Step 4: Replace outlier frames
    if len(outlier_frames) > 0:
        print(f"  Detected {len(outlier_frames)} outlier frames ({outlier_ratio*100:.1f}%)")
        print(f"  Attempting to fix using SLERP interpolation...")
        full_pose_quats, frames_fixed = replace_outlier_frames(full_pose_quats, outlier_frames)
        print(f"  Fixed {frames_fixed} frames")
    else:
        frames_fixed = 0
    
    # Step 5: Temporal smoothing (exclude fixed frames)
    full_pose_quats = smooth_quaternions_temporal(full_pose_quats, window_size, outlier_frames)
    
    # Step 6: Detect and smooth root translation outliers
    root_outlier_frames = detect_root_outliers(trans, threshold_cm_per_frame=50.0)
    if len(root_outlier_frames) > 0:
        print(f"  Detected {len(root_outlier_frames)} root translation outlier frames")
    
    # Smooth root translation (aggressive smoothing, especially for Z-axis/depth)
    # Use larger window (11) for better smoothing of oscillating motion
    trans_smoothed = smooth_root_translation(trans, window_size=11, outlier_frames=root_outlier_frames)
    
    # Apply velocity clamping to root translation (limit frame-to-frame displacement)
    # This prevents extreme jumps even after smoothing
    # Increased threshold for oscillating motion patterns
    max_root_velocity_cm_per_frame = 45.0  # Max 45 cm/frame displacement (increased for oscillating motion)
    trans_clamped = trans_smoothed.copy()
    root_velocities_clamped = 0
    
    for i in range(1, N):
        displacement = trans_smoothed[i] - trans_clamped[i-1]  # (3,) in meters
        displacement_cm = np.linalg.norm(displacement * 100.0)  # magnitude in cm
        
        if displacement_cm > max_root_velocity_cm_per_frame:
            # Scale down the displacement to max velocity
            scale = max_root_velocity_cm_per_frame / displacement_cm
            trans_clamped[i] = trans_clamped[i-1] + displacement * scale
            root_velocities_clamped += 1
    
    if root_velocities_clamped > 0:
        print(f"  Clamped {root_velocities_clamped} root translation velocities exceeding {max_root_velocity_cm_per_frame} cm/frame")
    
    trans_smoothed = trans_clamped
    
    # Step 7: Clamp angular velocity
    full_pose_quats, joints_clamped = clamp_angular_velocity(
        full_pose_quats, max_velocity, fps
    )
    
    # Compute max angular velocity after
    max_vel_after = compute_max_angular_velocity(full_pose_quats, fps)
    
    # Step 8: Convert back to axis-angle
    full_pose_aa = quaternion_to_axis_angle(full_pose_quats)  # (N, 25, 3)
    global_orient_cleaned = full_pose_aa[:, 0, :]  # (N, 3)
    pose_cleaned = full_pose_aa[:, 1:, :].reshape(N, 72)  # (N, 72)
    
    # Prepare output
    output_data = pkl_data.copy()
    if isinstance(output_data, dict):
        person_id = list(output_data.keys())[0]
        output_data[person_id]['pose'] = pose_cleaned
        output_data[person_id]['global_orient'] = global_orient_cleaned
        output_data[person_id]['trans'] = trans_smoothed
    else:
        output_data['pose'] = pose_cleaned
        output_data['global_orient'] = global_orient_cleaned
        output_data['trans'] = trans_smoothed
    
    # Save output
    save_pkl(output_data, output_pkl)
    
    # Generate report
    report = {
        'input_pkl': input_pkl,
        'output_pkl': output_pkl,
        'fps': fps,
        'total_frames': N,
        'outlier_detection': {
            'threshold_deg': outlier_threshold,
            'outlier_frames': sorted(list(outlier_frames)),
            'outlier_count': len(outlier_frames),
            'outlier_ratio': outlier_ratio,
            'frames_fixed': frames_fixed
        },
        'angular_velocity': {
            'max_before_deg_per_sec': max_vel_before,
            'max_after_deg_per_sec': max_vel_after,
            'clamp_threshold_deg_per_sec': max_velocity,
            'joints_clamped': joints_clamped
        },
        'smoothing': {
            'window_size': window_size,
            'root_translation_smoothed': True,
            'root_outlier_frames': sorted(list(root_outlier_frames)),
            'root_outlier_count': len(root_outlier_frames),
            'root_velocities_clamped': root_velocities_clamped,
            'max_root_velocity_cm_per_frame': max_root_velocity_cm_per_frame
        },
        'status': 'SUCCESS' if outlier_ratio <= MAX_OUTLIER_RATIO else 'FAILED'
    }
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description='Stabilize PKL motion data to eliminate single-frame jitter and rotation spikes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python motion_stabilizer.py --input input.pkl --output clean.pkl --fps 30
  python motion_stabilizer.py --input input.pkl --output clean.pkl --fps 30 --window 5 --max_velocity 200
  python motion_stabilizer.py --input input.pkl --output clean.pkl --fps 30 --outlier_threshold 180
        """
    )
    
    parser.add_argument('--input', type=str, required=True,
                        help='Input PKL file path')
    parser.add_argument('--output', type=str, required=True,
                        help='Output PKL file path')
    parser.add_argument('--fps', type=float, default=DEFAULT_FPS,
                        help=f'Frames per second (default: {DEFAULT_FPS})')
    parser.add_argument('--outlier_threshold', type=float, default=DEFAULT_OUTLIER_THRESHOLD_DEG,
                        help=f'Outlier detection threshold in degrees (default: {DEFAULT_OUTLIER_THRESHOLD_DEG})')
    parser.add_argument('--max_velocity', type=float, default=DEFAULT_MAX_ANGULAR_VELOCITY_DEG_PER_SEC,
                        help=f'Maximum angular velocity in deg/sec (default: {DEFAULT_MAX_ANGULAR_VELOCITY_DEG_PER_SEC})')
    parser.add_argument('--window', type=int, default=DEFAULT_SMOOTHING_WINDOW,
                        help=f'Temporal smoothing window size (default: {DEFAULT_SMOOTHING_WINDOW})')
    parser.add_argument('--report', type=str, default=None,
                        help='Output JSON report path (optional)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.isfile(args.input):
        print(f"ERROR: Input PKL file not found: {args.input}")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Run stabilization
    try:
        print("=" * 70)
        print("Motion Stabilization")
        print("=" * 70)
        print(f"Input: {args.input}")
        print(f"Output: {args.output}")
        print(f"FPS: {args.fps}")
        print(f"Outlier threshold: {args.outlier_threshold}°")
        print(f"Max velocity: {args.max_velocity}°/sec")
        print(f"Smoothing window: {args.window} frames")
        print()
        
        report = stabilize_motion(
            args.input, args.output, args.fps,
            args.outlier_threshold, args.max_velocity, args.window
        )
        
        # Print summary
        print("=" * 70)
        print("Stabilization Complete")
        print("=" * 70)
        print(f"Total frames: {report['total_frames']}")
        print(f"Outlier frames detected: {report['outlier_detection']['outlier_count']} "
              f"({report['outlier_detection']['outlier_ratio']*100:.1f}%)")
        print(f"Frames fixed: {report['outlier_detection']['frames_fixed']}")
        print(f"Joints clamped: {report['angular_velocity']['joints_clamped']}")
        print(f"Max angular velocity: {report['angular_velocity']['max_before_deg_per_sec']:.1f}°/sec -> "
              f"{report['angular_velocity']['max_after_deg_per_sec']:.1f}°/sec")
        print(f"Status: {report['status']}")
        print()
        
        if report['status'] == 'SUCCESS':
            print("✅ Motion is artist-ready")
        else:
            print("⚠️  Motion has many outliers but was processed")
            print("   Review output carefully - may need upstream fixes")
            # Don't exit with error - allow pipeline to continue
            # The motion validator will catch if it's truly unusable
        
        # Save report if requested
        if args.report:
            with open(args.report, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"Report saved to: {args.report}")
        
    except Exception as e:
        print(f"ERROR: Stabilization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
