#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validate SMPL output .pkl file
Checks: shapes, NaNs, Infs, axis-angle sanity
"""

import sys
import argparse
import numpy as np
import joblib
from loguru import logger


def validate_pkl(pkl_path):
    """
    Validate PKL output file
    """
    logger.info(f"Loading: {pkl_path}")
    try:
        data = joblib.load(pkl_path)
    except Exception as e:
        logger.error(f"Failed to load PKL: {e}")
        return False
    
    if not isinstance(data, dict):
        logger.error("PKL should contain a dictionary")
        return False
    
    all_valid = True
    
    for person_id, result in data.items():
        logger.info(f"\nValidating person {person_id}...")
        person_valid = True
        
        # Check required keys
        required_keys = ['pose', 'global_orient', 'betas', 'trans']
        for key in required_keys:
            if key not in result:
                logger.error(f"  Missing key: {key}")
                person_valid = False
                continue
            
            arr = result[key]
            
            # Check type
            if not isinstance(arr, np.ndarray):
                logger.error(f"  {key} is not a numpy array")
                person_valid = False
                continue
            
            # Check for NaNs
            if np.isnan(arr).any():
                logger.error(f"  {key} contains NaN values")
                person_valid = False
            
            # Check for Infs
            if np.isinf(arr).any():
                logger.error(f"  {key} contains Inf values")
                person_valid = False
            
            # Check shape
            if key == 'pose':
                if len(arr.shape) != 2 or arr.shape[1] != 72:
                    logger.error(f"  pose shape incorrect: expected (N, 72), got {arr.shape}")
                    person_valid = False
                else:
                    logger.info(f"  ✓ pose shape: {arr.shape}")
            elif key == 'global_orient':
                if len(arr.shape) != 2 or arr.shape[1] != 3:
                    logger.error(f"  global_orient shape incorrect: expected (N, 3), got {arr.shape}")
                    person_valid = False
                else:
                    logger.info(f"  ✓ global_orient shape: {arr.shape}")
            elif key == 'betas':
                if len(arr.shape) != 2 or arr.shape[1] != 10:
                    logger.error(f"  betas shape incorrect: expected (N, 10), got {arr.shape}")
                    person_valid = False
                else:
                    logger.info(f"  ✓ betas shape: {arr.shape}")
            elif key == 'trans':
                if len(arr.shape) != 2 or arr.shape[1] != 3:
                    logger.error(f"  trans shape incorrect: expected (N, 3), got {arr.shape}")
                    person_valid = False
                else:
                    logger.info(f"  ✓ trans shape: {arr.shape}")
                    # Check translation magnitude (should be reasonable)
                    trans_mag = np.linalg.norm(arr, axis=1)
                    max_trans = np.max(trans_mag)
                    mean_trans = np.mean(trans_mag)
                    logger.info(f"  ✓ trans magnitude: max={max_trans:.3f}, mean={mean_trans:.3f}")
        
        # Check optional keys
        if 'joints3d' in result:
            joints3d = result['joints3d']
            if isinstance(joints3d, np.ndarray):
                if len(joints3d.shape) != 3 or joints3d.shape[2] != 3:
                    logger.warning(f"  joints3d shape unexpected: {joints3d.shape} (expected (N, J, 3))")
                else:
                    logger.info(f"  ✓ joints3d shape: {joints3d.shape}")
        
        # Check axis-angle magnitude
        if 'pose' in result:
            pose = result['pose']
            # Reshape to (N*24, 3) for magnitude calculation
            pose_3d = pose.reshape(-1, 3)
            magnitudes = np.linalg.norm(pose_3d, axis=1)
            max_mag = np.max(magnitudes)
            mean_mag = np.mean(magnitudes)
            
            if max_mag > 10.0:
                logger.warning(f"  pose max axis-angle magnitude: {max_mag:.3f} (expected < 10.0)")
            else:
                logger.info(f"  ✓ pose axis-angle magnitude: max={max_mag:.3f}, mean={mean_mag:.3f}")
        
        if 'global_orient' in result:
            global_orient = result['global_orient']
            magnitudes = np.linalg.norm(global_orient, axis=1)
            max_mag = np.max(magnitudes)
            mean_mag = np.mean(magnitudes)
            
            if max_mag > 10.0:
                logger.warning(f"  global_orient max axis-angle magnitude: {max_mag:.3f} (expected < 10.0)")
            else:
                logger.info(f"  ✓ global_orient axis-angle magnitude: max={max_mag:.3f}, mean={mean_mag:.3f}")
        
        # Check frame_ids if present
        if 'frame_ids' in result:
            frame_ids = result['frame_ids']
            if isinstance(frame_ids, np.ndarray):
                logger.info(f"  ✓ frame_ids: {len(frame_ids)} frames")
        
        if person_valid:
            logger.info(f"  ✓ Person {person_id} validation PASSED")
        else:
            logger.error(f"  ✗ Person {person_id} validation FAILED")
            all_valid = False
    
    if all_valid:
        logger.info("\n" + "="*50)
        logger.info("✓ ALL VALIDATIONS PASSED")
        logger.info("="*50)
        return True
    else:
        logger.error("\n" + "="*50)
        logger.error("✗ VALIDATION FAILED")
        logger.error("="*50)
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Validate SMPL output .pkl file')
    parser.add_argument('pkl_file', type=str, help='Path to .pkl file to validate')
    
    args = parser.parse_args()
    
    success = validate_pkl(args.pkl_file)
    sys.exit(0 if success else 1)







