#!/usr/bin/env python
"""
Diagnostic script to check PKL file for accuracy issues
"""
import sys
import joblib
import numpy as np
from loguru import logger

def diagnose_pkl(pkl_path):
    """Run comprehensive diagnostics on a PKL file"""
    logger.info(f"Loading: {pkl_path}")
    data = joblib.load(pkl_path)
    
    issues = []
    warnings = []
    
    for person_id, person_data in data.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Diagnosing Person {person_id}")
        logger.info(f"{'='*60}")
        
        # Check 1: Required keys
        required_keys = ['pose', 'global_orient', 'betas', 'joints3d', 'frame_ids']
        missing_keys = [k for k in required_keys if k not in person_data]
        if missing_keys:
            issues.append(f"Person {person_id}: Missing keys: {missing_keys}")
            continue
        
        pose = person_data['pose']
        global_orient = person_data['global_orient']
        betas = person_data['betas']
        joints3d = person_data['joints3d']
        frame_ids = person_data['frame_ids']
        
        # Check 2: Shapes
        logger.info("\n1. Shape Checks:")
        expected_shapes = {
            'pose': (None, 72),
            'global_orient': (None, 3),
            'betas': (None, 10),
            'joints3d': (None, 49, 3),
        }
        
        n_frames = len(frame_ids)
        shape_ok = True
        if pose.shape != (n_frames, 72):
            issues.append(f"Person {person_id}: pose shape {pose.shape} != ({n_frames}, 72)")
            shape_ok = False
        if global_orient.shape != (n_frames, 3):
            issues.append(f"Person {person_id}: global_orient shape {global_orient.shape} != ({n_frames}, 3)")
            shape_ok = False
        if betas.shape != (n_frames, 10):
            issues.append(f"Person {person_id}: betas shape {betas.shape} != ({n_frames}, 10)")
            shape_ok = False
        if joints3d.shape != (n_frames, 49, 3):
            issues.append(f"Person {person_id}: joints3d shape {joints3d.shape} != ({n_frames}, 49, 3)")
            shape_ok = False
        
        if shape_ok:
            logger.info("   ✓ All shapes correct")
        
        # Check 3: Frame alignment
        logger.info("\n2. Frame Alignment:")
        if len(pose) != len(frame_ids):
            issues.append(f"Person {person_id}: Mismatch: {len(pose)} predictions vs {len(frame_ids)} frames")
        else:
            logger.info(f"   ✓ Frame count matches: {len(frame_ids)} frames")
        
        if not np.all(np.diff(frame_ids) > 0):
            issues.append(f"Person {person_id}: Frame IDs are not strictly increasing")
        else:
            logger.info("   ✓ Frame IDs are sequential")
        
        # Check 4: Global orient consistency
        logger.info("\n3. Global Orient Consistency:")
        pose_go = pose[:, :3]
        max_diff = np.max(np.abs(global_orient - pose_go))
        if max_diff > 1e-4:
            issues.append(f"Person {person_id}: Global orient mismatch: max_diff={max_diff:.6f}")
        else:
            logger.info(f"   ✓ Global orient matches pose[:3] (max_diff={max_diff:.6f})")
        
        # Check 5: NaN/Inf
        logger.info("\n4. Data Quality:")
        if np.isnan(pose).any():
            issues.append(f"Person {person_id}: pose contains NaN")
        else:
            logger.info("   ✓ No NaN in pose")
        
        if np.isinf(pose).any():
            issues.append(f"Person {person_id}: pose contains Inf")
        else:
            logger.info("   ✓ No Inf in pose")
        
        # Check 6: Axis-angle magnitude
        logger.info("\n5. Axis-Angle Magnitudes:")
        magnitudes = np.linalg.norm(pose.reshape(-1, 3), axis=1)
        max_mag = np.max(magnitudes)
        mean_mag = np.mean(magnitudes)
        logger.info(f"   Max magnitude: {max_mag:.4f} (should be < 2*pi ≈ 6.28)")
        logger.info(f"   Mean magnitude: {mean_mag:.4f}")
        
        if max_mag > 10.0:
            issues.append(f"Person {person_id}: Unusually large axis-angle magnitude: {max_mag:.4f}")
        elif max_mag > 6.5:
            warnings.append(f"Person {person_id}: Large axis-angle magnitude: {max_mag:.4f} (might indicate issues)")
        
        # Check 7: Temporal consistency
        logger.info("\n6. Temporal Consistency:")
        if len(pose) > 1:
            pose_diffs = np.linalg.norm(np.diff(pose, axis=0), axis=1)
            mean_diff = np.mean(pose_diffs)
            max_diff = np.max(pose_diffs)
            num_jumps = np.sum(pose_diffs > 2.0)  # Sudden jumps > 2.0
            
            logger.info(f"   Mean frame-to-frame change: {mean_diff:.4f}")
            logger.info(f"   Max frame-to-frame change: {max_diff:.4f}")
            logger.info(f"   Sudden jumps (>2.0): {num_jumps} frames ({100*num_jumps/len(pose_diffs):.1f}%)")
            
            if num_jumps > len(pose_diffs) * 0.2:  # More than 20% jumps
                warnings.append(f"Person {person_id}: Many sudden pose jumps ({num_jumps}/{len(pose_diffs)} frames)")
        
        # Check 8: Betas consistency (should be relatively stable)
        logger.info("\n7. Shape (Betas) Consistency:")
        if len(betas) > 1:
            beta_diffs = np.linalg.norm(np.diff(betas, axis=0), axis=1)
            mean_beta_diff = np.mean(beta_diffs)
            logger.info(f"   Mean frame-to-frame change: {mean_beta_diff:.4f}")
            if mean_beta_diff > 0.1:
                warnings.append(f"Person {person_id}: Betas changing significantly between frames (mean_diff={mean_beta_diff:.4f})")
        
        # Check 9: Joints3D validity
        logger.info("\n8. Joints3D Validity:")
        if np.isnan(joints3d).any():
            issues.append(f"Person {person_id}: joints3d contains NaN")
        else:
            logger.info("   ✓ No NaN in joints3d")
        
        # Check if joints are in reasonable range (typically -2 to 2 meters)
        joints_range = [joints3d.min(), joints3d.max()]
        logger.info(f"   Joints range: [{joints_range[0]:.2f}, {joints_range[1]:.2f}] meters")
        if abs(joints_range[0]) > 5 or abs(joints_range[1]) > 5:
            warnings.append(f"Person {person_id}: Joints3D in unusual range: {joints_range}")
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    
    if issues:
        logger.error(f"❌ Found {len(issues)} ISSUES:")
        for issue in issues:
            logger.error(f"   - {issue}")
    else:
        logger.info("✓ No critical issues found")
    
    if warnings:
        logger.warning(f"⚠ Found {len(warnings)} WARNINGS:")
        for warning in warnings:
            logger.warning(f"   - {warning}")
    else:
        logger.info("✓ No warnings")
    
    return len(issues) == 0


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python diagnose_pkl.py <pkl_file>")
        sys.exit(1)
    
    pkl_path = sys.argv[1]
    is_valid = diagnose_pkl(pkl_path)
    sys.exit(0 if is_valid else 1)



