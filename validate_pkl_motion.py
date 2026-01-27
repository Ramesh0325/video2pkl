#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Strict PKL Motion Validator for Maya Pipeline
Validates that root translation has actual motion (not constant/zero)
"""

import sys
import argparse
import numpy as np
import joblib
from loguru import logger


def validate_pkl_motion(pkl_path, min_delta_threshold=0.01):
    """
    Strictly validate that PKL contains meaningful root translation motion.
    
    Args:
        pkl_path: Path to PKL file
        min_delta_threshold: Minimum translation delta (in meters) to consider valid
    
    Returns:
        bool: True if valid, False otherwise
    """
    logger.info(f"Loading PKL: {pkl_path}")
    try:
        data = joblib.load(pkl_path)
    except Exception as e:
        logger.error(f"Failed to load PKL: {e}")
        return False
    
    if not isinstance(data, dict):
        logger.error("PKL should contain a dictionary")
        return False
    
    # Check if person 1 exists
    if 1 not in data:
        logger.error("Person 1 not found in PKL")
        return False
    
    result = data[1]
    
    # Check required keys
    required_keys = ['pose', 'trans']
    for key in required_keys:
        if key not in result:
            logger.error(f"Missing required key: {key}")
            return False
        
        arr = result[key]
        if not isinstance(arr, np.ndarray):
            logger.error(f"{key} is not a numpy array")
            return False
    
    # Validate pose shape
    pose = result['pose']
    if len(pose.shape) != 2 or pose.shape[1] != 72:
        logger.error(f"pose shape incorrect: expected (N, 72), got {pose.shape}")
        return False
    
    num_frames = pose.shape[0]
    logger.info(f"✓ Found {num_frames} frames")
    logger.info(f"✓ pose shape: {pose.shape}")
    
    # Validate trans shape
    trans = result['trans']
    if len(trans.shape) != 2 or trans.shape[1] != 3:
        logger.error(f"trans shape incorrect: expected (N, 3), got {trans.shape}")
        return False
    
    if trans.shape[0] != num_frames:
        logger.error(f"Frame count mismatch: pose has {num_frames} frames, trans has {trans.shape[0]}")
        return False
    
    logger.info(f"✓ trans shape: {trans.shape}")
    
    # Check for NaNs or Infs
    if np.isnan(trans).any():
        logger.error("trans contains NaN values")
        return False
    
    if np.isinf(trans).any():
        logger.error("trans contains Inf values")
        return False
    
    # Compute translation delta (per-axis range)
    trans_min = trans.min(axis=0)  # (3,)
    trans_max = trans.max(axis=0)  # (3,)
    trans_delta = trans_max - trans_min  # (3,)
    
    logger.info(f"\nTranslation Statistics:")
    logger.info(f"  X-axis: min={trans_min[0]:.6f}, max={trans_max[0]:.6f}, delta={trans_delta[0]:.6f}")
    logger.info(f"  Y-axis: min={trans_min[1]:.6f}, max={trans_max[1]:.6f}, delta={trans_delta[1]:.6f}")
    logger.info(f"  Z-axis: min={trans_min[2]:.6f}, max={trans_max[2]:.6f}, delta={trans_delta[2]:.6f}")
    
    # Check if translation has meaningful motion
    max_delta = np.max(trans_delta)
    
    if max_delta < min_delta_threshold:
        logger.error("\n" + "="*60)
        logger.error("❌ PKL VALIDATION FAILED")
        logger.error("="*60)
        logger.error(f"Translation delta: {trans_delta}")
        logger.error(f"Maximum delta: {max_delta:.6f} (threshold: {min_delta_threshold})")
        logger.error("\n❌ PKL is NOT usable for Maya (root locked)")
        logger.error("Root translation is constant or near-constant across all frames.")
        logger.error("Maya will see translation delta as [0, 0, 0]")
        logger.error("="*60)
        return False
    
    # Check if at least one axis has meaningful movement
    axes_with_motion = np.sum(trans_delta >= min_delta_threshold)
    
    if axes_with_motion == 0:
        logger.error("\n" + "="*60)
        logger.error("❌ PKL VALIDATION FAILED")
        logger.error("="*60)
        logger.error(f"Translation delta: {trans_delta}")
        logger.error("No axis has meaningful motion (all deltas < threshold)")
        logger.error("\n❌ PKL is NOT usable for Maya (root locked)")
        logger.error("="*60)
        return False
    
    # Success!
    logger.info("\n" + "="*60)
    logger.info("✅ PKL VALIDATION PASSED")
    logger.info("="*60)
    logger.info(f"Translation delta: {trans_delta}")
    logger.info(f"Maximum delta: {max_delta:.6f}")
    logger.info(f"Axes with motion: {axes_with_motion}/3")
    logger.info("\n✅ PKL is VALID for Maya (root motion present)")
    logger.info("="*60)
    
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Strict PKL Motion Validator - Ensures root translation has actual motion'
    )
    parser.add_argument('pkl_file', type=str, help='Path to .pkl file to validate')
    parser.add_argument(
        '--min_delta',
        type=float,
        default=0.01,
        help='Minimum translation delta (meters) to consider valid (default: 0.01)'
    )
    
    args = parser.parse_args()
    
    success = validate_pkl_motion(args.pkl_file, min_delta_threshold=args.min_delta)
    sys.exit(0 if success else 1)

