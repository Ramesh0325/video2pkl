#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SmoothNet Wrapper for PARE Pipeline

This module provides a wrapper around SmoothNet to smooth 3D joint positions
temporally, reducing jitter in PARE output before PKL generation.

SmoothNet is a plug-and-play temporal-only network that refines human poses
by learning long-range temporal relations without spatial modeling.
"""

import os
import sys
import torch
import numpy as np
from loguru import logger

# Add smoothnet to path
SMOOTHNET_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'smoothnet')
if SMOOTHNET_ROOT not in sys.path:
    sys.path.insert(0, SMOOTHNET_ROOT)

try:
    from lib.models.smoothnet import SmoothNet
    from lib.utils.utils import slide_window_to_sequence
except ImportError as e:
    logger.error(f"Failed to import SmoothNet: {e}")
    logger.error(f"SmoothNet root: {SMOOTHNET_ROOT}")
    raise


def load_smoothnet_model(checkpoint_path, window_size=32, device='cuda', 
                         hidden_size=512, res_hidden_size=16, num_blocks=1, dropout=0.5):
    """
    Load a pretrained SmoothNet model from checkpoint.
    
    Args:
        checkpoint_path: Path to SmoothNet checkpoint (.pth.tar file)
        window_size: Window size used during training (8, 16, 32, or 64)
        device: Device to load model on ('cuda' or 'cpu')
        hidden_size: Hidden size in encoder/decoder (default: 512)
        res_hidden_size: Hidden size in residual blocks (default: 16 for pw3d_spin_3D)
        num_blocks: Number of residual blocks (default: 1 for pw3d_spin_3D)
        dropout: Dropout probability (default: 0.5)
    
    Returns:
        model: Loaded SmoothNet model in eval mode
    """
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"SmoothNet checkpoint not found: {checkpoint_path}")
    
    # Create model with specified architecture parameters
    # Defaults match pw3d_spin_3D config: hidden_size=512, res_hidden_size=16, num_blocks=1
    model = SmoothNet(
        window_size=window_size,
        output_size=window_size,  # Output same size as input
        hidden_size=hidden_size,
        res_hidden_size=res_hidden_size,
        num_blocks=num_blocks,
        dropout=dropout
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
        logger.info(f"Loaded SmoothNet from {checkpoint_path} (performance: {checkpoint.get('performance', 'N/A')})")
    else:
        # Assume checkpoint is just the state dict
        model.load_state_dict(checkpoint)
        logger.info(f"Loaded SmoothNet from {checkpoint_path}")
    
    model.eval()
    return model


def apply_smoothnet_to_joints(joints3d, model, device='cuda', window_step=1,
                                exclude_root=True, motion_critical_only=True,
                                max_velocity=2.0):
    """
    Apply SmoothNet smoothing to 3D joint positions with safety constraints.
    
    CRITICAL FIXES:
    1. Excludes root joint (joint 0) from SmoothNet to prevent global pops
    2. Only smooths motion-critical joints (spine, shoulders, arms, legs)
    3. Applies velocity clamping to prevent unrealistic motion spikes
    4. Preserves original root and excluded joints
    
    Args:
        joints3d: numpy array of shape (T, J, 3) where T=num_frames, J=num_joints (49 for SMPL)
        model: Loaded SmoothNet model
        device: Device to run inference on ('cuda' or 'cpu')
        window_step: Step size for sliding window (default 1)
        exclude_root: If True, exclude root joint (joint 0) from smoothing
        motion_critical_only: If True, only smooth motion-critical joints
        max_velocity: Maximum allowed velocity in m/s per frame (default 2.0)
    
    Returns:
        smoothed_joints3d: numpy array of same shape (T, J, 3) with smoothed joints
    """
    # Convert to torch tensor if needed
    if isinstance(joints3d, np.ndarray):
        joints3d_tensor = torch.from_numpy(joints3d).float()
        return_numpy = True
    else:
        joints3d_tensor = joints3d.float()
        return_numpy = False
    
    T, J, D = joints3d_tensor.shape
    assert D == 3, f"Expected 3D joints, got shape {joints3d_tensor.shape}"
    
    # Start with original joints - we'll only modify selected joints
    smoothed_joints3d = joints3d_tensor.clone()
    
    # Identify joints to smooth
    # Root joint is always index 0 (SMPL hips/pelvis)
    ROOT_JOINT_IDX = 0
    
    # Motion-critical joints in 49-joint SMPL array:
    # Based on SMPL structure: 24 base joints + 25 extra joints
    # Motion-critical: spine (3,6,9), shoulders (13,14), arms (16-21), legs (1,2,4,5,7,8), neck (12)
    # In 49-joint array, first 24 are SMPL base joints, so indices match for base joints
    if motion_critical_only:
        # SMPL base joint indices (0-23) that are motion-critical
        # Exclude: root (0), head (15), hands (20,21), feet (7,8), toes (10,11)
        # Include: spine (3,6,9), shoulders (13,14), arms (16,17,18,19), legs (1,2,4,5), neck (12)
        motion_critical_smpl_indices = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 16, 17, 18, 19]
        # For 49-joint array, first 24 indices are SMPL base joints
        joints_to_smooth = [idx for idx in motion_critical_smpl_indices if idx < J]
        if exclude_root and ROOT_JOINT_IDX in joints_to_smooth:
            joints_to_smooth.remove(ROOT_JOINT_IDX)
    else:
        # Smooth all joints except root
        joints_to_smooth = list(range(J))
        if exclude_root:
            joints_to_smooth.remove(ROOT_JOINT_IDX)
    
    if len(joints_to_smooth) == 0:
        logger.warning("No joints selected for smoothing, returning original")
        if return_numpy:
            return joints3d_tensor.cpu().numpy()
        return joints3d_tensor
    
    logger.info(f"Smoothing {len(joints_to_smooth)}/{J} joints (excluding root and non-critical)")
    
    # Extract only joints to smooth
    joints_to_smooth_tensor = joints3d_tensor[:, joints_to_smooth, :]  # (T, num_smooth_joints, 3)
    num_smooth_joints = len(joints_to_smooth)
    
    window_size = model.window_size
    
    # Handle sequences shorter than window size
    if T < window_size:
        logger.warning(f"Sequence length ({T}) is shorter than window size ({window_size}). "
                      f"Padding with edge frames.")
        pad_before = (window_size - T) // 2
        pad_after = window_size - T - pad_before
        first_frame = joints_to_smooth_tensor[0:1].repeat(pad_before, 1, 1)
        last_frame = joints_to_smooth_tensor[-1:].repeat(pad_after, 1, 1)
        joints_to_smooth_tensor = torch.cat([first_frame, joints_to_smooth_tensor, last_frame], dim=0)
        T_padded = joints_to_smooth_tensor.shape[0]
        was_padded = True
    else:
        T_padded = T
        was_padded = False
    
    # Reshape from (T, num_smooth_joints, 3) to (T, num_smooth_joints*3) for SmoothNet
    joints_flat = joints_to_smooth_tensor.reshape(T_padded, num_smooth_joints * 3)  # (T, num_smooth_joints*3)
    
    # Create sliding windows
    windows = []
    for i in range(0, T_padded - window_size + 1, window_step):
        window = joints_flat[i:i + window_size]  # (window_size, num_smooth_joints*3)
        windows.append(window)
    
    if len(windows) == 0:
        # Edge case: exactly window_size frames
        windows = [joints_flat]
    
    # Stack windows: (num_windows, window_size, num_smooth_joints*3)
    windows_tensor = torch.stack(windows)
    
    # Permute to (num_windows, num_smooth_joints*3, window_size) for SmoothNet
    windows_tensor = windows_tensor.permute(0, 2, 1)  # (N, C, T)
    
    # Move to device
    windows_tensor = windows_tensor.to(device)
    
    # Run SmoothNet inference
    with torch.no_grad():
        smoothed_windows = model(windows_tensor)  # (N, C, window_size)
    
    # Permute back: (N, window_size, C)
    smoothed_windows = smoothed_windows.permute(0, 2, 1)  # (N, window_size, num_smooth_joints*3)
    
    # Convert sliding windows back to full sequence
    smoothed_sequence = slide_window_to_sequence(
        smoothed_windows,
        window_step=window_step,
        window_size=window_size
    )  # (output_len, num_smooth_joints*3)
    
    # Reshape back to (T, num_smooth_joints, 3)
    smoothed_selected_joints = smoothed_sequence.reshape(-1, num_smooth_joints, 3)
    
    # Remove padding if we added it
    if was_padded:
        smoothed_selected_joints = smoothed_selected_joints[pad_before:T_padded-pad_after]
    
    # Apply velocity clamping to prevent unrealistic spikes
    logger.info("Applying velocity clamping to prevent motion spikes...")
    smoothed_selected_joints_np = smoothed_selected_joints.cpu().numpy()
    
    # Compute velocities (frame-to-frame differences)
    velocities = np.diff(smoothed_selected_joints_np, axis=0)  # (T-1, num_smooth_joints, 3)
    velocity_magnitudes = np.linalg.norm(velocities, axis=2)  # (T-1, num_smooth_joints)
    
    # Clamp velocities that exceed threshold
    # max_velocity is in m/s, assuming ~30fps: 2.0 m/s ≈ 0.067 m/frame
    # Use a more conservative threshold: 0.1 m/frame (≈3 m/s at 30fps)
    max_vel_per_frame = max_velocity / 30.0  # Convert m/s to m/frame (assuming 30fps)
    clamped_frames = 0
    
    # Clamp velocities that exceed threshold
    for t in range(len(velocities)):
        for j in range(num_smooth_joints):
            if velocity_magnitudes[t, j] > max_vel_per_frame:
                # Clamp velocity to max
                scale = max_vel_per_frame / (velocity_magnitudes[t, j] + 1e-8)
                velocities[t, j] *= scale
                clamped_frames += 1
    
    if clamped_frames > 0:
        logger.warning(f"Clamped {clamped_frames} joint velocities exceeding {max_vel_per_frame:.4f} m/frame "
                      f"({max_velocity} m/s at 30fps)")
        # Reconstruct positions from clamped velocities
        smoothed_selected_joints_np_clamped = smoothed_selected_joints_np.copy()
        smoothed_selected_joints_np_clamped[0] = smoothed_selected_joints_np[0]  # Keep first frame
        for t in range(1, len(smoothed_selected_joints_np_clamped)):
            smoothed_selected_joints_np_clamped[t] = smoothed_selected_joints_np_clamped[t-1] + velocities[t-1]
        smoothed_selected_joints_np = smoothed_selected_joints_np_clamped
    
    # Insert smoothed joints back into full joint array
    for i, joint_idx in enumerate(joints_to_smooth):
        smoothed_joints3d[:, joint_idx, :] = torch.from_numpy(smoothed_selected_joints_np[:, i, :]).float()
    
    # Root joint and excluded joints remain unchanged (preserved from original)
    
    # Convert back to numpy if input was numpy
    if return_numpy:
        smoothed_joints3d = smoothed_joints3d.cpu().numpy()
    
    return smoothed_joints3d


def smooth_joints3d_with_smoothnet(joints3d, checkpoint_path, window_size=32, device='cuda'):
    """
    High-level function to smooth 3D joints using SmoothNet.
    
    This function loads the model, applies smoothing, and returns smoothed joints.
    For repeated calls, consider loading the model once and using
    apply_smoothnet_to_joints directly.
    
    Args:
        joints3d: numpy array of shape (T, J, 3) - 3D joint positions
        checkpoint_path: Path to SmoothNet checkpoint file
        window_size: Window size used during training (must match checkpoint)
        device: Device to run on ('cuda' or 'cpu')
    
    Returns:
        smoothed_joints3d: numpy array of same shape (T, J, 3) with smoothed joints
    """
    # Load model
    model = load_smoothnet_model(checkpoint_path, window_size=window_size, device=device)
    
    # Apply smoothing
    smoothed_joints3d = apply_smoothnet_to_joints(joints3d, model, device=device)
    
    return smoothed_joints3d
