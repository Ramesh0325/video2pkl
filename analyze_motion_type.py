#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Motion Type Analyzer

Analyzes PKL motion to determine if it's:
- Progressing (forward/backward movement)
- Looping (repeating motion)
- Stationary (minimal movement)
"""

import argparse
import numpy as np
from typing import Dict, Tuple

try:
    import joblib
    USE_JOBLIB = True
except ImportError:
    import pickle
    USE_JOBLIB = False


def load_pkl(pkl_path: str) -> Dict:
    """Load PKL file (supports both joblib and pickle)."""
    if USE_JOBLIB:
        return joblib.load(pkl_path)
    else:
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)


def extract_translation(pkl_data: Dict) -> np.ndarray:
    """Extract root translation from PKL."""
    if isinstance(pkl_data, dict):
        person_id = list(pkl_data.keys())[0]
        return pkl_data[person_id]['trans']  # (N, 3)
    return pkl_data['trans']


def analyze_motion_type(trans: np.ndarray, fps: float = 30.0) -> Dict:
    """
    Analyze motion to determine if it's progressing, looping, or stationary.
    
    Args:
        trans: (N, 3) root translation
        fps: Frames per second
    
    Returns:
        Dictionary with analysis results
    """
    N = trans.shape[0]
    
    # Compute frame-to-frame displacements
    displacements = np.diff(trans, axis=0)  # (N-1, 3) in meters
    magnitudes = np.linalg.norm(displacements, axis=1)  # (N-1,) in meters
    
    # Convert to cm
    magnitudes_cm = magnitudes * 100.0
    
    # Analyze per-axis movement
    x_movement = np.abs(displacements[:, 0] * 100.0)  # cm
    y_movement = np.abs(displacements[:, 1] * 100.0)  # cm
    z_movement = np.abs(displacements[:, 2] * 100.0)  # cm
    
    # Total displacement (start to end)
    total_displacement = np.linalg.norm(trans[-1] - trans[0]) * 100.0  # cm
    
    # Cumulative path length (total distance traveled)
    cumulative_path = np.sum(magnitudes_cm)  # cm
    
    # Direction analysis
    # Check if motion has a dominant direction (progressing) vs oscillating (looping)
    x_direction_changes = np.sum(np.diff(np.sign(displacements[:, 0])) != 0)
    y_direction_changes = np.sum(np.diff(np.sign(displacements[:, 1])) != 0)
    z_direction_changes = np.sum(np.diff(np.sign(displacements[:, 2])) != 0)
    
    # Net displacement vs path length ratio
    # High ratio (>0.5) = progressing, Low ratio (<0.3) = looping/oscillating
    if cumulative_path > 0:
        progress_ratio = total_displacement / cumulative_path
    else:
        progress_ratio = 0.0
    
    # Check for periodicity (looping)
    # Compare first half vs second half
    mid_point = N // 2
    first_half = trans[:mid_point]
    second_half = trans[mid_point:]
    
    # Normalize by subtracting initial position
    first_half_norm = first_half - first_half[0]
    second_half_norm = second_half - second_half[0]
    
    # Resample second half to match first half length
    if len(second_half_norm) > len(first_half_norm):
        indices = np.linspace(0, len(second_half_norm) - 1, len(first_half_norm)).astype(int)
        second_half_resampled = second_half_norm[indices]
    else:
        second_half_resampled = second_half_norm
    
    # Compute similarity (correlation)
    if len(first_half_norm) == len(second_half_resampled):
        # Flatten and compute correlation
        first_flat = first_half_norm.flatten()
        second_flat = second_half_resampled.flatten()
        if np.std(first_flat) > 0 and np.std(second_flat) > 0:
            correlation = np.corrcoef(first_flat, second_flat)[0, 1]
        else:
            correlation = 0.0
    else:
        correlation = 0.0
    
    # Determine motion type
    if cumulative_path < 10.0:  # Less than 10 cm total movement
        motion_type = "STATIONARY"
    elif progress_ratio > 0.5 and total_displacement > 50.0:
        motion_type = "PROGRESSING"
    elif correlation > 0.7:
        motion_type = "LOOPING"
    elif progress_ratio < 0.3:
        motion_type = "OSCILLATING"
    else:
        motion_type = "MIXED"
    
    # Dominant axis
    axis_movements = [np.sum(x_movement), np.sum(y_movement), np.sum(z_movement)]
    dominant_axis = ['X', 'Y', 'Z'][np.argmax(axis_movements)]
    
    return {
        'motion_type': motion_type,
        'total_frames': N,
        'duration_sec': N / fps,
        'total_displacement_cm': total_displacement,
        'cumulative_path_cm': cumulative_path,
        'progress_ratio': progress_ratio,
        'correlation_first_second_half': correlation,
        'direction_changes': {
            'X': x_direction_changes,
            'Y': y_direction_changes,
            'Z': z_direction_changes
        },
        'axis_movements_cm': {
            'X': np.sum(x_movement),
            'Y': np.sum(y_movement),
            'Z': np.sum(z_movement)
        },
        'max_displacement_per_frame_cm': np.max(magnitudes_cm),
        'mean_displacement_per_frame_cm': np.mean(magnitudes_cm),
        'dominant_axis': dominant_axis,
        'z_axis_range_cm': {
            'min': np.min(trans[:, 2]) * 100.0,
            'max': np.max(trans[:, 2]) * 100.0,
            'delta': (np.max(trans[:, 2]) - np.min(trans[:, 2])) * 100.0
        }
    }


def print_analysis(report: Dict):
    """Print analysis results in a readable format."""
    print("\n" + "=" * 70)
    print("MOTION TYPE ANALYSIS")
    print("=" * 70)
    print(f"\nMotion Type: {report['motion_type']}")
    print(f"Total Frames: {report['total_frames']}")
    print(f"Duration: {report['duration_sec']:.2f} seconds")
    
    print(f"\n--- Displacement Analysis ---")
    print(f"Total Displacement (start to end): {report['total_displacement_cm']:.2f} cm")
    print(f"Cumulative Path Length: {report['cumulative_path_cm']:.2f} cm")
    print(f"Progress Ratio (displacement/path): {report['progress_ratio']:.3f}")
    print(f"  → {'High' if report['progress_ratio'] > 0.5 else 'Low'} ratio suggests {'progressing' if report['progress_ratio'] > 0.5 else 'looping/oscillating'} motion")
    
    print(f"\n--- Per-Axis Movement ---")
    for axis, movement in report['axis_movements_cm'].items():
        print(f"  {axis}-axis: {movement:.2f} cm total")
    print(f"Dominant Axis: {report['dominant_axis']}")
    
    print(f"\n--- Z-Axis (Depth) Analysis ---")
    z_range = report['z_axis_range_cm']
    print(f"  Range: {z_range['min']:.2f} to {z_range['max']:.2f} cm")
    print(f"  Delta: {z_range['delta']:.2f} cm")
    print(f"  → {'Large' if z_range['delta'] > 1000 else 'Moderate' if z_range['delta'] > 100 else 'Small'} depth change")
    
    print(f"\n--- Direction Changes ---")
    for axis, changes in report['direction_changes'].items():
        print(f"  {axis}-axis: {changes} direction changes")
    
    print(f"\n--- Frame-to-Frame Movement ---")
    print(f"  Max displacement: {report['max_displacement_per_frame_cm']:.2f} cm/frame")
    print(f"  Mean displacement: {report['mean_displacement_per_frame_cm']:.2f} cm/frame")
    
    if report['correlation_first_second_half'] > 0:
        print(f"\n--- Periodicity Check ---")
        print(f"  Correlation (first half vs second half): {report['correlation_first_second_half']:.3f}")
        if report['correlation_first_second_half'] > 0.7:
            print(f"  → High correlation suggests LOOPING motion")
        elif report['correlation_first_second_half'] < 0.3:
            print(f"  → Low correlation suggests PROGRESSING motion")
    
    print("\n" + "=" * 70)
    
    # Interpretation
    print("\nINTERPRETATION:")
    if report['motion_type'] == "PROGRESSING":
        print("  ✓ Motion is PROGRESSING (forward/backward movement)")
        print("    Large Z-axis changes are expected for depth movement")
        print("    High outlier ratio may be due to rapid depth changes")
    elif report['motion_type'] == "LOOPING":
        print("  ✓ Motion is LOOPING (repeating pattern)")
        print("    Motion should be more stable - high outliers suggest jitter")
    elif report['motion_type'] == "OSCILLATING":
        print("  ✓ Motion is OSCILLATING (back and forth)")
        print("    Many direction changes are expected")
    elif report['motion_type'] == "STATIONARY":
        print("  ✓ Motion is mostly STATIONARY")
        print("    High outliers suggest camera jitter or tracking issues")
    else:
        print("  ? Motion type is MIXED")
        print("    Combination of progressing and oscillating")
    
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze PKL motion to determine if it\'s progressing, looping, or stationary',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--pkl', type=str, required=True,
                        help='Input PKL file path')
    parser.add_argument('--fps', type=float, default=30.0,
                        help='Frames per second (default: 30.0)')
    
    args = parser.parse_args()
    
    # Load PKL
    print(f"Loading PKL: {args.pkl}")
    pkl_data = load_pkl(args.pkl)
    
    # Extract translation
    trans = extract_translation(pkl_data)
    print(f"Found {trans.shape[0]} frames")
    
    # Analyze
    report = analyze_motion_type(trans, args.fps)
    
    # Print results
    print_analysis(report)


if __name__ == '__main__':
    main()
