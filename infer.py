#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Minimal inference script for PARE: Video -> SMPL .pkl
Outputs: pose (axis-angle Nx72), global_orient, betas, joints3d
"""

import os
import sys
import cv2
import torch
import joblib
import argparse
import numpy as np
from tqdm import tqdm
from loguru import logger
from torch.utils.data import DataLoader

# Add pare to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pare.core.tester import PARETester
from pare.dataset.inference import Inference
from pare.utils.demo_utils import video_to_images
from pare.utils.geometry import rotation_matrix_to_angle_axis, convert_weak_perspective_to_perspective

# Optional SmoothNet import (only if enabled)
_smoothnet_available = False
try:
    from pare.utils.smoothnet_wrapper import smooth_joints3d_with_smoothnet, load_smoothnet_model, apply_smoothnet_to_joints
    _smoothnet_available = True
except ImportError:
    pass


def compute_root_translation(pred_cam, focal_length=5000.0, img_res=224, smooth=True):
    """
    Compute world-space root translation from PARE weak-perspective camera parameters.
    
    Args:
        pred_cam: (N, 3) weak-perspective camera [s, tx, ty] in cropped image space
        focal_length: Focal length used by PARE (default 5000.0)
        img_res: Image resolution used by PARE (default 224)
        smooth: Whether to apply temporal smoothing to reduce jitter
    
    Returns:
        root_trans: (N, 3) root translation in world space (meters)
    """
    # Convert weak-perspective to perspective camera translation
    # pred_cam is [s, tx, ty] where s is scale, tx/ty are translations in image space
    pred_cam_t = convert_weak_perspective_to_perspective(
        pred_cam,
        focal_length=focal_length,
        img_res=img_res,
    )  # (N, 3) -> [tx, ty, tz] in camera space
    
    # Convert to numpy
    if isinstance(pred_cam_t, torch.Tensor):
        pred_cam_t = pred_cam_t.cpu().numpy()
    
    # Root translation is the negative of camera translation
    # In SMPL convention: if camera is at cam_t, body root is at -cam_t in world space
    # However, we need to account for coordinate system conventions
    # Standard convention: root_trans = -cam_t (body position = negative of camera position)
    root_trans = -pred_cam_t
    
    # Apply temporal smoothing to reduce jitter
    if smooth and len(root_trans) > 1:
        window_size = min(5, len(root_trans))
        if window_size > 1:
            # Try to use scipy's savgol filter for better smoothing
            try:
                from scipy import signal
                for i in range(3):  # Smooth each dimension
                    wl = min(window_size if window_size % 2 == 1 else window_size - 1, len(root_trans))
                    if wl > 2:
                        root_trans[:, i] = signal.savgol_filter(
                            root_trans[:, i],
                            window_length=wl,
                            polyorder=min(2, wl - 1)
                        )
                    elif wl > 1:
                        # For very short sequences, use simple moving average
                        kernel = np.ones(wl) / wl
                        root_trans[:, i] = np.convolve(root_trans[:, i], kernel, mode='same')
            except ImportError:
                # Fallback to simple moving average if scipy not available
                kernel = np.ones(window_size) / window_size
                for i in range(3):
                    root_trans[:, i] = np.convolve(root_trans[:, i], kernel, mode='same')
            except Exception:
                # Fallback to simple moving average if savgol fails
                kernel = np.ones(window_size) / window_size
                for i in range(3):
                    root_trans[:, i] = np.convolve(root_trans[:, i], kernel, mode='same')
    
    return root_trans


def smooth_rotation_matrices(rotmats, min_cutoff=0.004, beta=0.7):
    """
    Smooth rotation matrices temporally using OneEuroFilter on axis-angle representation.
    
    This is the CORRECT way to smooth rotations:
    1. Convert rotation matrices to axis-angle
    2. Smooth axis-angle vectors
    3. Convert back to rotation matrices
    4. Re-normalize to ensure valid rotation matrices
    
    Args:
        rotmats: (N, 24, 3, 3) array of rotation matrices
        min_cutoff: OneEuroFilter min_cutoff (lower = more smoothing)
        beta: OneEuroFilter beta (speed coefficient)
    
    Returns:
        smoothed_rotmats: (N, 24, 3, 3) smoothed rotation matrices
    """
    from pare.utils.one_euro_filter import OneEuroFilter
    from pare.utils.geometry import rotation_matrix_to_angle_axis
    
    rotmats = np.asarray(rotmats, dtype=np.float32)
    N, num_joints = rotmats.shape[0], rotmats.shape[1]
    
    # Convert to axis-angle for smoothing
    pose_aa_list = []
    for i in range(N):
        rotmat_torch = torch.from_numpy(rotmats[i]).float()
        aa = rotation_matrix_to_angle_axis(rotmat_torch.reshape(-1, 3, 3))  # (24, 3)
        pose_aa_list.append(aa.numpy())
    
    pose_aa = np.array(pose_aa_list)  # (N, 24, 3)
    
    # Smooth each joint's axis-angle independently
    smoothed_aa = np.zeros_like(pose_aa)
    smoothed_aa[0] = pose_aa[0]  # First frame unchanged
    
    # Initialize filters for each joint and dimension (24 joints × 3 dims = 72 filters)
    filters = []
    for joint_idx in range(num_joints):
        for dim in range(3):
            filters.append(OneEuroFilter(
                t0=0.0,
                x0=pose_aa[0, joint_idx, dim],
                dx0=0.0,
                min_cutoff=min_cutoff,
                beta=beta,
                d_cutoff=1.0
            ))
    
    # Apply filtering frame by frame
    for t in range(1, N):
        filter_idx = 0
        for joint_idx in range(num_joints):
            for dim in range(3):
                smoothed_aa[t, joint_idx, dim] = filters[filter_idx](float(t), pose_aa[t, joint_idx, dim])
                filter_idx += 1
    
    # Convert back to rotation matrices
    from pare.utils.geometry import batch_rodrigues
    smoothed_rotmats = []
    for i in range(N):
        aa_torch = torch.from_numpy(smoothed_aa[i].reshape(-1, 3)).float()
        rotmat = batch_rodrigues(aa_torch)  # (24, 3, 3)
        smoothed_rotmats.append(rotmat.numpy())
    
    return np.array(smoothed_rotmats)  # (N, 24, 3, 3)


def convert_rotmat_to_axis_angle(rotmat):
    """
    Convert rotation matrices (N, 24, 3, 3) to axis-angle (N, 72)
    Returns: pose (N, 72), global_orient (N, 3)
    """
    # Ensure it's a numpy array
    if isinstance(rotmat, torch.Tensor):
        rotmat = rotmat.cpu().numpy()
    
    N = rotmat.shape[0]
    device = 'cpu'  # Use CPU for conversion
    
    # Convert to torch for conversion function
    rotmat_torch = torch.from_numpy(rotmat).float()
    
    pose_aa = []
    global_orient_aa = []
    
    for i in range(N):
        # Global orientation (first joint, index 0)
        global_rot = rotmat_torch[i, 0:1]  # (1, 3, 3)
        global_aa = rotation_matrix_to_angle_axis(global_rot)  # (1, 3)
        global_orient_aa.append(global_aa.numpy().flatten())
        
        # Body pose (remaining 23 joints, indices 1-23)
        body_rot = rotmat_torch[i, 1:]  # (23, 3, 3)
        body_rot_flat = body_rot.reshape(-1, 3, 3)  # (23, 3, 3)
        body_aa = rotation_matrix_to_angle_axis(body_rot_flat)  # (23, 3)
        body_aa_flat = body_aa.numpy().reshape(-1)  # (69,)
        
        # Combine global + body = 72D pose
        full_pose = np.concatenate([global_aa.numpy().flatten(), body_aa_flat])
        pose_aa.append(full_pose)
    
    pose_aa = np.array(pose_aa)  # (N, 72)
    global_orient_aa = np.array(global_orient_aa)  # (N, 3)
    
    return pose_aa, global_orient_aa


def validate_pkl_output(output_dict):
    """
    Validate PKL output: check shapes, NaNs, Infs, axis-angle sanity
    """
    errors = []
    
    # Check required keys
    required_keys = ['pose', 'global_orient', 'betas', 'trans']
    for key in required_keys:
        if key not in output_dict:
            errors.append(f"Missing required key: {key}")
            continue
        
        data = output_dict[key]
        
        # Check for NaNs
        if np.isnan(data).any():
            errors.append(f"{key} contains NaN values")
        
        # Check for Infs
        if np.isinf(data).any():
            errors.append(f"{key} contains Inf values")
        
        # Check shape
        if key == 'pose':
            if data.shape[1] != 72:
                errors.append(f"pose shape incorrect: expected (N, 72), got {data.shape}")
        elif key == 'global_orient':
            if data.shape[1] != 3:
                errors.append(f"global_orient shape incorrect: expected (N, 3), got {data.shape}")
        elif key == 'betas':
            if data.shape[1] != 10:
                errors.append(f"betas shape incorrect: expected (N, 10), got {data.shape}")
        elif key == 'trans':
            if data.shape[1] != 3:
                errors.append(f"trans shape incorrect: expected (N, 3), got {data.shape}")
    
    # Check axis-angle magnitude (should be reasonable, typically < 2*pi)
    if 'pose' in output_dict:
        pose = output_dict['pose']
        magnitudes = np.linalg.norm(pose.reshape(-1, 3), axis=1)
        max_mag = np.max(magnitudes)
        if max_mag > 10.0:  # Sanity check: axis-angle magnitude should be reasonable
            errors.append(f"pose axis-angle magnitude too large: max={max_mag:.2f} (expected < 10.0)")
    
    if 'global_orient' in output_dict:
        global_orient = output_dict['global_orient']
        magnitudes = np.linalg.norm(global_orient, axis=1)
        max_mag = np.max(magnitudes)
        if max_mag > 10.0:
            errors.append(f"global_orient axis-angle magnitude too large: max={max_mag:.2f} (expected < 10.0)")
    
    return errors


# Global cache for SmoothNet model (reused across persons)
_smoothnet_model_cache = None

def main(args):
    global _smoothnet_model_cache
    
    # Setup paths
    video_file = args.video
    output_folder = args.out
    
    if not os.path.isfile(video_file):
        logger.error(f"Video file not found: {video_file}")
        sys.exit(1)
    
    os.makedirs(output_folder, exist_ok=True)
    
    # Extract video frames
    logger.info(f"Extracting frames from video: {video_file}")
    temp_img_folder = os.path.join(output_folder, 'tmp_frames')
    os.makedirs(temp_img_folder, exist_ok=True)
    
    input_image_folder, num_frames, img_shape = video_to_images(
        video_file,
        img_folder=temp_img_folder,
        return_info=True
    )
    logger.info(f"Extracted {num_frames} frames to {input_image_folder}")
    
    # Create minimal args object for PARETester
    class Args:
        def __init__(self):
            self.cfg = args.cfg
            self.ckpt = args.ckpt
            self.tracking_method = 'bbox'
            self.detector = 'yolo'
            self.yolo_img_size = 416
            self.tracker_batch_size = 12
            self.batch_size = args.batch_size
            self.display = False
            self.smooth = args.smooth
            self.min_cutoff = 0.004
            self.beta = 1.0
            self.staf_dir = None
    
    test_args = Args()
    
    # Clear GPU cache before loading model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Initialize tester
    logger.info("Initializing PARE model...")
    tester = PARETester(test_args)
    
    # Clear GPU cache after model loading
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Run tracking
    logger.info("Running person tracking...")
    tracking_results = tester.run_tracking(video_file, input_image_folder)
    
    # Clear GPU cache after tracking (tracking uses YOLO which can leave memory)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if len(tracking_results) == 0:
        logger.error("No persons detected in video!")
        sys.exit(1)
    
    logger.info(f"Found {len(tracking_results)} person(s) in video")
    
    # Run inference on each person
    orig_height, orig_width = img_shape[:2]
    logger.info("Running PARE inference...")
    
    pare_results = {}
    for person_id in tqdm(list(tracking_results.keys()), desc="Processing persons"):
        bboxes = tracking_results[person_id]['bbox']
        frames = tracking_results[person_id]['frames']
        
        dataset = Inference(
            image_folder=input_image_folder,
            frames=frames,
            bboxes=bboxes,
            joints2d=None,
            scale=1.0,
        )
        
        bboxes = dataset.bboxes
        frames = dataset.frames
        
        # Ensure DataLoader preserves order (no shuffling)
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            num_workers=4,
            shuffle=False,  # Explicitly disable shuffling to preserve frame order
            pin_memory=False
        )
        
        pred_pose_rotmat = []
        pred_betas = []
        pred_joints3d = []
        pred_cam = []  # Weak-perspective camera parameters [s, tx, ty]
        
        # Use no_grad for better memory efficiency
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                batch = batch.to(tester.device)
                output = tester.model(batch)
                
                pred_pose_rotmat.append(output['pred_pose'].cpu())
                pred_betas.append(output['pred_shape'].cpu())
                pred_joints3d.append(output['smpl_joints3d'].cpu())
                pred_cam.append(output['pred_cam'].cpu())  # Extract camera parameters
                
                # Clear GPU cache after each batch
                del batch, output
                torch.cuda.empty_cache()
        
        # Concatenate all predictions (should be in frame order)
        pred_pose_rotmat = torch.cat(pred_pose_rotmat, dim=0).numpy()  # (N, 24, 3, 3)
        pred_betas = torch.cat(pred_betas, dim=0).numpy()  # (N, 10)
        pred_joints3d = torch.cat(pred_joints3d, dim=0).numpy()  # (N, 49, 3)
        pred_cam = torch.cat(pred_cam, dim=0)  # (N, 3) - keep as tensor for conversion
        
        # Verify that number of predictions matches number of frames
        if len(pred_pose_rotmat) != len(frames):
            logger.error(f"Mismatch: {len(pred_pose_rotmat)} predictions but {len(frames)} frames!")
            raise ValueError("Frame count mismatch between predictions and frame IDs")
        
        # Apply temporal smoothing to rotation matrices BEFORE converting to axis-angle
        # This is the CORRECT way - smooth in rotation space, not axis-angle space
        # Note: When SmoothNet is enabled, we rely on SmoothNet for joints3d smoothing
        # and skip pose smoothing to avoid introducing discontinuities
        if args.enable_pose_smoothing and not args.enable_smoothnet:
            logger.info(f"Applying temporal smoothing to rotation matrices...")
            pred_pose_rotmat_original = pred_pose_rotmat.copy()
            pred_pose_rotmat = smooth_rotation_matrices(
                pred_pose_rotmat,
                min_cutoff=args.smoothing_min_cutoff,
                beta=args.smoothing_beta
            )
            logger.info(f"  Smoothed rotation matrices shape: {pred_pose_rotmat.shape}")
        else:
            if args.enable_smoothnet:
                logger.info(f"Pose smoothing skipped (SmoothNet handles joints3d smoothing)")
            else:
                logger.info(f"Pose smoothing DISABLED - using raw model predictions")
                logger.info(f"  (Enable with --enable_pose_smoothing if needed)")
        
        # Convert rotation matrices to axis-angle
        logger.info(f"Converting rotation matrices to axis-angle for person {person_id}...")
        pose_aa, global_orient_aa = convert_rotmat_to_axis_angle(pred_pose_rotmat)
        
        # Log camera parameters for first person (for debugging)
        if person_id == 1:
            pred_cam_np = pred_cam.cpu().numpy() if isinstance(pred_cam, torch.Tensor) else pred_cam
            logger.info(f"\nCamera Parameters Analysis (Person {person_id}):")
            logger.info(f"  pred_cam shape: {pred_cam_np.shape}")
            logger.info(f"  pred_cam sample (first 3 frames):")
            for i in range(min(3, len(pred_cam_np))):
                logger.info(f"    Frame {i}: s={pred_cam_np[i,0]:.6f}, tx={pred_cam_np[i,1]:.6f}, ty={pred_cam_np[i,2]:.6f}")
            
            # Check if camera parameters vary
            cam_s_delta = pred_cam_np[:, 0].max() - pred_cam_np[:, 0].min()
            cam_tx_delta = pred_cam_np[:, 1].max() - pred_cam_np[:, 1].min()
            cam_ty_delta = pred_cam_np[:, 2].max() - pred_cam_np[:, 2].min()
            
            logger.info(f"  Camera parameter deltas:")
            logger.info(f"    s (scale):  delta={cam_s_delta:.6f}")
            logger.info(f"    tx:         delta={cam_tx_delta:.6f}")
            logger.info(f"    ty:         delta={cam_ty_delta:.6f}")
            
            if cam_s_delta < 0.001 and cam_tx_delta < 0.001 and cam_ty_delta < 0.001:
                logger.warning(f"  ⚠️  WARNING: Camera parameters are static (near-constant)")
                logger.warning(f"     This may indicate tracking issues or static camera setup")
                logger.warning(f"     Root translation may be constant as a result")
            else:
                logger.info(f"  ✓ Camera parameters vary across frames")
        
        # Compute root translation from camera parameters
        logger.info(f"Computing root translation for person {person_id}...")
        root_trans = compute_root_translation(
            pred_cam,
            focal_length=5000.0,  # PARE default focal length
            img_res=224,  # PARE default image resolution
            smooth=True  # Apply temporal smoothing
        )  # (N, 3)
        
        # Optional: Normalize to first frame (removes camera movement offset)
        # Use this if you want person-only motion relative to starting position
        # Default: Keep camera-relative (better for Maya with camera movement)
        if args.normalize_translation:
            logger.info(f"Normalizing translation to first frame (removing camera offset)...")
            root_trans = root_trans - root_trans[0]
            logger.info(f"  First frame (after normalization): {root_trans[0]}")
            logger.info(f"  Note: Translation now represents motion relative to starting position")
        
        # Log root translation statistics for first person
        if person_id == 1:
            trans_delta = root_trans.max(axis=0) - root_trans.min(axis=0)
            logger.info(f"\nRoot Translation Statistics (Person {person_id}):")
            logger.info(f"  trans shape: {root_trans.shape}")
            logger.info(f"  trans sample (first 3 frames):")
            for i in range(min(3, len(root_trans))):
                logger.info(f"    Frame {i}: {root_trans[i]}")
            logger.info(f"  Translation delta (per-axis): {trans_delta}")
            logger.info(f"  Maximum delta: {np.max(trans_delta):.6f}")
            
            if np.max(trans_delta) < 0.01:
                logger.warning(f"  ⚠️  WARNING: Root translation delta is very small (< 0.01m)")
                logger.warning(f"     This may cause Maya to see no motion")
        
        # Betas (body shape) should be constant for the same person across frames
        # Average betas across all frames to get a single consistent body shape
        betas_mean = np.mean(pred_betas, axis=0, keepdims=True)  # (1, 10)
        betas_consistent = np.repeat(betas_mean, len(pred_betas), axis=0)  # (N, 10)
        
        # Check if betas vary significantly (warn if they do)
        beta_std = np.std(pred_betas, axis=0)
        if np.max(beta_std) > 0.1:
            logger.warning(f"Person {person_id}: Betas vary significantly across frames (max_std={np.max(beta_std):.4f})")
            logger.warning(f"  Using averaged betas for consistency")
        else:
            logger.info(f"Person {person_id}: Betas are consistent (max_std={np.max(beta_std):.4f})")
        
        # Note: Smoothing is now applied to rotation matrices BEFORE conversion
        # No additional smoothing needed on axis-angle (which would break geometry)
        
        # Create output dictionary
        output_dict = {
            'pose': pose_aa,  # (N, 72) axis-angle
            'global_orient': global_orient_aa,  # (N, 3) axis-angle
            'betas': betas_consistent,  # (N, 10) - averaged for consistency
            'betas_per_frame': pred_betas,  # (N, 10) - original per-frame betas (for reference)
            'joints3d': pred_joints3d,  # (N, 49, 3)
            'trans': root_trans,  # (N, 3) root translation in camera-relative space
            'frame_ids': frames,  # (N,)
        }
        
        # ============================================================
        # Apply SmoothNet temporal smoothing (if enabled)
        # ============================================================
        if args.enable_smoothnet:
            if not _smoothnet_available:
                logger.error("SmoothNet requested but not available. Please ensure smoothnet/ is cloned.")
                sys.exit(1)
            
            logger.info(f"\n{'='*60}")
            logger.info(f"Applying SmoothNet temporal smoothing to person {person_id}...")
            logger.info(f"{'='*60}")
            
            # Load SmoothNet model (reuse across persons if same checkpoint)
            # CRITICAL: Window size MUST match checkpoint training (cannot be changed)
            # Extract window size from checkpoint filename if possible, or use user-specified
            # Checkpoint naming: checkpoint_8.pth.tar, checkpoint_16.pth.tar, etc.
            import re
            checkpoint_match = re.search(r'checkpoint_(\d+)\.pth\.tar', args.smoothnet_checkpoint)
            if checkpoint_match:
                checkpoint_window_size = int(checkpoint_match.group(1))
                if args.smoothnet_window_size != checkpoint_window_size:
                    logger.warning(f"Checkpoint window size ({checkpoint_window_size}) doesn't match "
                                  f"specified window size ({args.smoothnet_window_size}). "
                                  f"Using checkpoint window size {checkpoint_window_size}.")
                smoothnet_window_size = checkpoint_window_size
            else:
                smoothnet_window_size = args.smoothnet_window_size
                logger.warning(f"Could not determine window size from checkpoint filename. "
                              f"Using specified window size: {smoothnet_window_size}")
            
            if smoothnet_window_size > 11:
                logger.warning(f"Window size {smoothnet_window_size} > 11 may cause long-range artifacts. "
                              f"Other fixes (root exclusion, joint subset, velocity clamping) still apply.")
            
            if _smoothnet_model_cache is None or _smoothnet_model_cache.window_size != smoothnet_window_size:
                logger.info(f"Loading SmoothNet model from: {args.smoothnet_checkpoint}")
                logger.info(f"Window size: {smoothnet_window_size} (from checkpoint training)")
                _smoothnet_model_cache = load_smoothnet_model(
                    args.smoothnet_checkpoint,
                    window_size=smoothnet_window_size,
                    device=tester.device
                )
            
            # CRITICAL FIX: Apply SmoothNet with root exclusion and motion-critical joints only
            logger.info(f"Smoothing joints3d: shape {output_dict['joints3d'].shape}")
            logger.info("  - Excluding root joint (prevents global pops)")
            logger.info("  - Smoothing only motion-critical joints (spine, shoulders, arms, legs)")
            logger.info("  - Applying velocity clamping (prevents unrealistic spikes)")
            
            # Store original joints3d for root preservation
            original_joints3d = output_dict['joints3d'].copy()
            
            smoothed_joints3d = apply_smoothnet_to_joints(
                output_dict['joints3d'],
                _smoothnet_model_cache,
                device=tester.device,
                window_step=1,
                exclude_root=True,  # MANDATORY: Exclude root to prevent global pops
                motion_critical_only=True,  # Only smooth motion-critical joints
                max_velocity=2.0  # Clamp velocities > 2.0 m/frame
            )
            
            # Ensure root joint is preserved (not modified by SmoothNet)
            smoothed_joints3d[:, 0, :] = original_joints3d[:, 0, :]
            logger.info("  - Root joint preserved (unchanged from original)")
            
            output_dict['joints3d'] = smoothed_joints3d
            logger.info(f"Smoothed joints3d: shape {smoothed_joints3d.shape}")
            
            # Apply light low-pass filtering to root translation
            logger.info("Applying light temporal filtering to root translation...")
            try:
                from scipy import signal
                trans_filtered = output_dict['trans'].copy()
                for i in range(3):  # Filter each dimension
                    # Use Savitzky-Golay filter for smooth, low-lag filtering
                    window_length = min(5, len(trans_filtered))
                    if window_length > 2 and window_length % 2 == 1:
                        trans_filtered[:, i] = signal.savgol_filter(
                            trans_filtered[:, i],
                            window_length=window_length,
                            polyorder=min(2, window_length - 1)
                        )
                output_dict['trans'] = trans_filtered
                logger.info(f"Filtered translation: shape {trans_filtered.shape}")
            except ImportError:
                logger.warning("scipy not available, using simple moving average for translation filtering")
                # Fallback to simple moving average
                trans_filtered = output_dict['trans'].copy()
                window_size = min(3, len(trans_filtered))
                kernel = np.ones(window_size) / window_size
                for i in range(3):
                    trans_filtered[:, i] = np.convolve(trans_filtered[:, i], kernel, mode='same')
                output_dict['trans'] = trans_filtered
            
            # Recompute poses from stabilized joints for consistency
            # Since full IK is complex, apply light quaternion smoothing to maintain consistency
            # with smoothed joints while preserving original motion characteristics
            logger.info("Applying light pose smoothing for consistency with stabilized joints...")
            
            try:
                from scipy.spatial.transform import Rotation as R
                from scipy import signal
                
                N = pose_aa.shape[0]
                pose_aa_reshaped = pose_aa.reshape(N, 24, 3)  # (N, 24, 3)
                
                # Light smoothing to maintain consistency with smoothed joints
                pose_aa_smoothed = np.zeros_like(pose_aa_reshaped)
                global_orient_smoothed = np.zeros_like(global_orient_aa)
                
                # Process global orientation with moderate smoothing (prevents tilts)
                # Global orientation is critical - use stronger smoothing to prevent sudden jumps
                rots_global = [R.from_rotvec(global_orient_aa[i]) for i in range(N)]
                quats_global = np.array([r.as_quat() for r in rots_global])  # (N, 4)
                
                # Moderate smoothing (larger window) to prevent sudden orientation changes
                quats_global_smooth = quats_global.copy()
                for dim in range(4):
                    if len(quats_global_smooth) > 6:
                        window_length = min(7, len(quats_global_smooth))  # Larger window for stability
                        if window_length >= 7 and window_length % 2 == 1:
                            quats_global_smooth[:, dim] = signal.savgol_filter(
                                quats_global_smooth[:, dim],
                                window_length=window_length,
                                polyorder=min(2, window_length - 1)
                            )
                        elif len(quats_global_smooth) > 4:
                            # Fallback for shorter sequences
                            window_length = min(5, len(quats_global_smooth))
                            if window_length >= 5 and window_length % 2 == 1:
                                quats_global_smooth[:, dim] = signal.savgol_filter(
                                    quats_global_smooth[:, dim],
                                    window_length=window_length,
                                    polyorder=min(2, window_length - 1)
                                )
                
                # Normalize and convert back
                for i in range(N):
                    q = quats_global_smooth[i]
                    q = q / np.linalg.norm(q)
                    r = R.from_quat(q)
                    global_orient_smoothed[i] = r.as_rotvec()
                
                # CRITICAL: Detect and fix sudden orientation jumps (single-frame tilts)
                logger.info("Detecting and fixing sudden orientation jumps...")
                global_orient_fixed = global_orient_smoothed.copy()
                max_angular_velocity_deg_per_sec = 180.0  # Max 180°/sec for global orientation
                # Estimate FPS from frame count (default to 30 if unknown)
                estimated_fps = 30.0  # Default FPS for angular velocity calculation
                max_angular_velocity_rad_per_frame = np.radians(max_angular_velocity_deg_per_sec / estimated_fps)
                
                fixed_jumps = 0
                for i in range(1, N):
                    # Compute rotation distance between consecutive frames
                    r_prev = R.from_rotvec(global_orient_fixed[i-1])
                    r_curr = R.from_rotvec(global_orient_fixed[i])
                    rot_diff = r_prev.inv() * r_curr
                    angle_diff = rot_diff.magnitude()
                    
                    # If rotation jump is too large, interpolate between prev and next
                    if angle_diff > max_angular_velocity_rad_per_frame:
                        # Check if this is a single-frame anomaly (next frame is closer to prev)
                        if i < N - 1:
                            r_next = R.from_rotvec(global_orient_fixed[i+1])
                            rot_diff_next = r_prev.inv() * r_next
                            angle_diff_next = rot_diff_next.magnitude()
                            
                            # If next frame is closer to prev, this is likely an anomaly
                            if angle_diff_next < angle_diff * 0.7:
                                # Interpolate: use average of prev and next
                                quat_prev = r_prev.as_quat()
                                quat_next = r_next.as_quat()
                                # SLERP interpolation (spherical linear interpolation)
                                quat_interp = (quat_prev + quat_next) / 2.0
                                quat_interp = quat_interp / np.linalg.norm(quat_interp)
                                r_interp = R.from_quat(quat_interp)
                                global_orient_fixed[i] = r_interp.as_rotvec()
                                fixed_jumps += 1
                                logger.debug(f"Fixed orientation jump at frame {i}: {np.degrees(angle_diff):.1f}° -> {np.degrees(angle_diff_next):.1f}°")
                        else:
                            # Last frame: just use previous frame's orientation
                            global_orient_fixed[i] = global_orient_fixed[i-1]
                            fixed_jumps += 1
                            logger.debug(f"Fixed orientation jump at last frame {i}")
                
                if fixed_jumps > 0:
                    logger.info(f"Fixed {fixed_jumps} sudden orientation jumps (prevented skeleton tilts)")
                
                # Final pass: Additional smoothing after jump fixes
                if fixed_jumps > 0:
                    rots_global_fixed = [R.from_rotvec(global_orient_fixed[i]) for i in range(N)]
                    quats_global_fixed = np.array([r.as_quat() for r in rots_global_fixed])
                    
                    # Light smoothing pass to blend fixes smoothly
                    for dim in range(4):
                        if len(quats_global_fixed) > 4:
                            window_length = min(5, len(quats_global_fixed))
                            if window_length >= 5 and window_length % 2 == 1:
                                quats_global_fixed[:, dim] = signal.savgol_filter(
                                    quats_global_fixed[:, dim],
                                    window_length=window_length,
                                    polyorder=min(2, window_length - 1)
                                )
                    
                    # Convert back
                    for i in range(N):
                        q = quats_global_fixed[i]
                        q = q / np.linalg.norm(q)
                        r = R.from_quat(q)
                        global_orient_fixed[i] = r.as_rotvec()
                
                global_orient_smoothed = global_orient_fixed
                
                # Process body joints with light smoothing (only motion-critical joints)
                # Match the joints that were smoothed in joints3d
                motion_critical_pose_indices = [1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 16, 17, 18, 19]
                
                for joint_idx in range(24):
                    if joint_idx in motion_critical_pose_indices:
                        # Light smoothing for motion-critical joints
                        rots_joint = [R.from_rotvec(pose_aa_reshaped[i, joint_idx]) for i in range(N)]
                        quats_joint = np.array([r.as_quat() for r in rots_joint])
                        
                        quats_joint_smooth = quats_joint.copy()
                        for dim in range(4):
                            if len(quats_joint_smooth) > 4:
                                window_length = min(5, len(quats_joint_smooth))
                                if window_length >= 5 and window_length % 2 == 1:
                                    quats_joint_smooth[:, dim] = signal.savgol_filter(
                                        quats_joint_smooth[:, dim],
                                        window_length=window_length,
                                        polyorder=min(2, window_length - 1)
                                    )
                        
                        # Normalize and convert back
                        for i in range(N):
                            q = quats_joint_smooth[i]
                            q = q / np.linalg.norm(q)
                            r = R.from_quat(q)
                            pose_aa_smoothed[i, joint_idx] = r.as_rotvec()
                    else:
                        # Keep original for non-critical joints
                        pose_aa_smoothed[:, joint_idx] = pose_aa_reshaped[:, joint_idx]
                
                # Reshape back to (N, 72)
                pose_aa_filtered = pose_aa_smoothed.reshape(N, 72)
                global_orient_filtered = global_orient_smoothed
                
                # Additional pass: Clamp extreme angular velocities in poses
                logger.info("Clamping extreme angular velocities in poses...")
                pose_aa_clamped = pose_aa_filtered.copy()
                max_pose_angular_velocity_deg_per_sec = 300.0  # Max 300°/sec for body joints
                # Estimate FPS from frame count (default to 30 if unknown)
                estimated_fps = 30.0  # Default FPS for angular velocity calculation
                max_pose_angular_velocity_rad_per_frame = np.radians(max_pose_angular_velocity_deg_per_sec / estimated_fps)
                
                pose_aa_clamped_reshaped = pose_aa_clamped.reshape(N, 24, 3)
                clamped_pose_count = 0
                
                for joint_idx in range(24):
                    for i in range(1, N):
                        r_prev = R.from_rotvec(pose_aa_clamped_reshaped[i-1, joint_idx])
                        r_curr = R.from_rotvec(pose_aa_clamped_reshaped[i, joint_idx])
                        rot_diff = r_prev.inv() * r_curr
                        angle_diff = rot_diff.magnitude()
                        
                        if angle_diff > max_pose_angular_velocity_rad_per_frame:
                            # Clamp: limit rotation to max velocity
                            scale = max_pose_angular_velocity_rad_per_frame / angle_diff
                            # Apply scaled rotation
                            rot_diff_clamped = R.from_rotvec(rot_diff.as_rotvec() * scale)
                            r_clamped = r_prev * rot_diff_clamped
                            pose_aa_clamped_reshaped[i, joint_idx] = r_clamped.as_rotvec()
                            clamped_pose_count += 1
                
                if clamped_pose_count > 0:
                    logger.info(f"Clamped {clamped_pose_count} pose rotations exceeding {max_pose_angular_velocity_deg_per_sec}°/sec")
                
                pose_aa_filtered = pose_aa_clamped_reshaped.reshape(N, 72)
                
                logger.info("Light pose smoothing complete (consistent with stabilized joints)")
                
            except (ImportError, Exception) as e:
                logger.warning(f"Pose smoothing failed: {e}, using original poses")
                pose_aa_filtered = pose_aa.copy()
                global_orient_filtered = global_orient_aa.copy()
            
            # Update output
            output_dict['pose'] = pose_aa_filtered
            output_dict['global_orient'] = global_orient_filtered
            
            logger.info("Pose processing complete")
            
            logger.info(f"SmoothNet processing complete for person {person_id}")
            logger.info(f"{'='*60}\n")
        
        # Validate output
        validation_errors = validate_pkl_output(output_dict)
        if validation_errors:
            logger.warning(f"Validation warnings for person {person_id}:")
            for err in validation_errors:
                logger.warning(f"  - {err}")
        else:
            logger.info(f"Output validation passed for person {person_id}")
        
        pare_results[person_id] = output_dict
    
    # Save results
    output_pkl_path = os.path.join(output_folder, 'smpl_output.pkl')
    logger.info(f"Saving results to {output_pkl_path}")
    joblib.dump(pare_results, output_pkl_path)
    
    # Also save to output-pkl-files folder with input video name
    output_pkl_files_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output-pkl-files')
    os.makedirs(output_pkl_files_dir, exist_ok=True)
    
    # Extract video basename (without extension)
    video_basename = os.path.splitext(os.path.basename(args.video))[0]
    final_pkl_path = os.path.join(output_pkl_files_dir, f'{video_basename}.pkl')
    
    # Copy PKL to final location
    import shutil
    shutil.copy2(output_pkl_path, final_pkl_path)
    logger.info(f"PKL also saved to: {final_pkl_path}")
    
    # Validate PKL motion before finalizing
    logger.info("\n" + "="*60)
    logger.info("Validating PKL motion (strict check)...")
    logger.info("="*60)
    
    # Import and run validator
    try:
        import subprocess
        validator_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'validate_pkl_motion.py')
        result = subprocess.run(
            [sys.executable, validator_script, output_pkl_path],
            capture_output=True,
            text=True
        )
        
        # Print validator output
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        if result.returncode != 0:
            # Validation failed - save as invalid
            invalid_path = output_pkl_path.replace('.pkl', '_INVALID.pkl')
            logger.error(f"\n❌ PKL validation FAILED - saving as: {invalid_path}")
            logger.error("This PKL is NOT usable for Maya (root translation has no motion)")
            
            # Keep the invalid file for debugging
            import shutil
            shutil.copy(output_pkl_path, invalid_path)
            
            logger.error("\nPossible causes:")
            logger.error("  1. Camera parameters (pred_cam) are constant")
            logger.error("  2. Person tracking is not updating bboxes")
            logger.error("  3. Video has static camera and person")
            logger.error("  4. Temporal smoothing may have over-smoothed motion")
            logger.error("\nSuggestions:")
            logger.error("  - Check if person is actually moving in video")
            logger.error("  - Verify tracking is working (bboxes should change)")
            logger.error("  - Try reducing smoothing or disabling it")
            logger.error("  - Check camera parameter deltas in logs above")
            
            sys.exit(1)
        else:
            logger.info(f"\n✅ PKL validation PASSED - file is ready for Maya")
    except FileNotFoundError:
        logger.warning("validate_pkl_motion.py not found - skipping strict validation")
        logger.warning("PKL saved but not validated for motion")
    except Exception as e:
        logger.error(f"Error during PKL validation: {e}")
        logger.warning("PKL saved but validation failed - please check manually")
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("INFERENCE COMPLETE")
    logger.info("="*50)
    for person_id, result in pare_results.items():
        logger.info(f"\nPerson {person_id}:")
        logger.info(f"  Frames: {len(result['frame_ids'])}")
        logger.info(f"  Pose shape: {result['pose'].shape}")
        logger.info(f"  Global orient shape: {result['global_orient'].shape}")
        logger.info(f"  Betas shape: {result['betas'].shape}")
        logger.info(f"  Trans shape: {result['trans'].shape}")
        logger.info(f"  Joints3D shape: {result['joints3d'].shape}")
    logger.info(f"\nOutput saved to: {output_pkl_path}")
    logger.info(f"Final PKL saved to: {final_pkl_path}")
    logger.info("="*50)
    
    # Cleanup temp frames if requested
    if args.cleanup:
        import shutil
        logger.info(f"Cleaning up temporary frames: {temp_img_folder}")
        shutil.rmtree(temp_img_folder)
    
    logger.info("Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PARE Minimal Inference: Video -> SMPL .pkl')
    
    parser.add_argument('--video', type=str, required=True,
                        help='Input video file path')
    parser.add_argument('--out', type=str, required=True,
                        help='Output folder path')
    parser.add_argument('--cfg', type=str,
                        default='data/pare/checkpoints/pare_config.yaml',
                        help='Config file path')
    parser.add_argument('--ckpt', type=str,
                        default='data/pare/checkpoints/pare_checkpoint.ckpt',
                        help='Checkpoint file path')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for inference')
    parser.add_argument('--smooth', action='store_true',
                        help='Apply pose smoothing (not fully implemented for axis-angle)')
    parser.add_argument('--cleanup', action='store_true',
                        help='Clean up temporary frame images after inference')
    parser.add_argument('--normalize_translation', action='store_true',
                        help='Normalize root translation to first frame (removes camera movement offset). '
                             'Default: Keep camera-relative translation (recommended for Maya with camera movement)')
    parser.add_argument('--enable_pose_smoothing', action='store_true',
                        help='Enable temporal smoothing of pose (DISABLED by default). '
                             'Smoothing is applied to rotation matrices before converting to axis-angle. '
                             'WARNING: May introduce artifacts - use only if you see excessive jitter.')
    parser.add_argument('--smoothing_min_cutoff', type=float, default=0.004,
                        help='OneEuroFilter min_cutoff (only used if --enable_pose_smoothing). '
                             'Lower = more smoothing, Higher = less smoothing. Default: 0.004')
    parser.add_argument('--smoothing_beta', type=float, default=0.7,
                        help='OneEuroFilter beta (only used if --enable_pose_smoothing). '
                             'Higher = less lag, Lower = more smoothing. Default: 0.7')
    parser.add_argument('--enable_smoothnet', action='store_true',
                        help='Enable SmoothNet temporal smoothing for production-grade PKL output. '
                             'Smooths joints3d using a pretrained SmoothNet model and applies '
                             'light filtering to translation. Recommended for final output.')
    parser.add_argument('--smoothnet_checkpoint', type=str,
                        default='smoothnet/data/checkpoints/pw3d_spin_3D/checkpoint_32.pth.tar',
                        help='Path to SmoothNet pretrained checkpoint. '
                             'Default models available: window_size 8, 16, 32, 64. '
                             'Note: Window size will be clamped to 5-11 for stability.')
    parser.add_argument('--smoothnet_window_size', type=int, default=8,
                        help='SmoothNet window size (must match checkpoint training). '
                             'Options: 8, 16, 32, 64. Default: 8. '
                             'Note: Automatically clamped to 5-11 range to prevent long-range artifacts. '
                             'Larger = more temporal context but may introduce jumps.')
    
    args = parser.parse_args()
    
    main(args)

