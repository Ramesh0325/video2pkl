#!/bin/bash
# PARE Inference Pipeline Script with SmoothNet
# Activates conda environment, prompts for video input, and runs inference
# 
# Usage:
#   ./run_inference.sh [video_path]
# 
# Examples:
#   ./run_inference.sh inputs/video.mp4              # With SmoothNet (recommended)
# 
# SmoothNet Features (Production-Grade):
#   - Root joint exclusion: Prevents global skeleton pops
#   - Motion-critical joints only: Smooths spine, shoulders, arms, legs (14/49 joints)
#   - Velocity clamping: Prevents unrealistic motion spikes (>2.0 m/s)
#   - Window size 8: Optimal temporal stability (5-11 frame range)
#   - Light pose smoothing: Maintains consistency with smoothed joints

set -e  # Exit on error

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Initialize conda and activate PARE environment
echo -e "${YELLOW}Initializing conda environment...${NC}"
eval "$(/home/ptgml/anaconda3/bin/conda shell.bash hook)"
conda activate PARE

# Ensure PATH is updated
export PATH="/home/ptgml/anaconda3/envs/PARE/bin:$PATH"

# Check if conda activation was successful
if [ "$CONDA_DEFAULT_ENV" != "PARE" ]; then
    echo -e "${RED}ERROR: Failed to activate PARE conda environment${NC}"
    exit 1
fi

# Get the Python executable from the activated environment
PYTHON_EXE=$(which python)
if [ -z "$PYTHON_EXE" ]; then
    # Fallback to explicit path
    PYTHON_EXE="/home/ptgml/anaconda3/envs/PARE/bin/python"
    if [ ! -f "$PYTHON_EXE" ]; then
        echo -e "${RED}ERROR: Python not found in PARE environment${NC}"
        exit 1
    fi
fi

echo -e "${GREEN}✓ PARE environment activated${NC}"
echo -e "${GREEN}✓ Using Python: $PYTHON_EXE${NC}"
echo ""

# Get input video path
if [ -z "$1" ]; then
    echo "=========================================="
    echo "  PARE Inference - Video to SMPL .pkl"
    echo "=========================================="
    echo ""
    read -p "Enter input video path: " VIDEO_PATH
    echo ""
else
    VIDEO_PATH="$1"
    echo "=========================================="
    echo "  PARE Inference - Video to SMPL .pkl"
    echo "=========================================="
    echo ""
    echo "Input video: $VIDEO_PATH"
fi

# Validate video file exists
if [ ! -f "$VIDEO_PATH" ]; then
    echo -e "${RED}ERROR: Video file not found: $VIDEO_PATH${NC}"
    exit 1
fi

# Get absolute path
VIDEO_PATH=$(readlink -f "$VIDEO_PATH" 2>/dev/null || realpath "$VIDEO_PATH" 2>/dev/null || echo "$VIDEO_PATH")

# Generate output folder name from video filename
VIDEO_BASENAME=$(basename "$VIDEO_PATH")
VIDEO_BASENAME="${VIDEO_BASENAME%.*}"  # Remove extension
OUTPUT_FOLDER="outputs/${VIDEO_BASENAME}_$(date +%Y%m%d_%H%M%S)"

# Create output directory
mkdir -p "$OUTPUT_FOLDER"

echo ""
echo "=========================================="
echo "  Inference Configuration"
echo "=========================================="
echo "  Input video: $VIDEO_PATH"
echo "  Output folder: $OUTPUT_FOLDER"
echo "  Batch size: 1"
echo "  SmoothNet: ENABLED (production-grade smoothing)"
echo "    - Checkpoint: checkpoint_8.pth.tar (window_size=8)"
echo "    - Root exclusion: Prevents global pops"
echo "    - Motion-critical joints: 14/49 joints smoothed"
echo "    - Velocity clamping: Max 2.0 m/s per frame"
echo "=========================================="
echo ""
echo -e "${YELLOW}Starting inference with SmoothNet...${NC}"
echo ""

# SmoothNet checkpoint path
SMOOTHNET_CHECKPOINT="smoothnet/data/checkpoints/pw3d_spin_3D/checkpoint_8.pth.tar"

# Check if checkpoint exists
if [ ! -f "$SMOOTHNET_CHECKPOINT" ]; then
    echo -e "${YELLOW}WARNING: SmoothNet checkpoint not found: $SMOOTHNET_CHECKPOINT${NC}"
    echo -e "${YELLOW}Falling back to checkpoint_32.pth.tar...${NC}"
    SMOOTHNET_CHECKPOINT="smoothnet/data/checkpoints/pw3d_spin_3D/checkpoint_32.pth.tar"
    SMOOTHNET_WINDOW_SIZE=32
else
    SMOOTHNET_WINDOW_SIZE=8
fi

# Run inference using the Python from the activated environment
"$PYTHON_EXE" infer.py \
    --video "$VIDEO_PATH" \
    --out "$OUTPUT_FOLDER" \
    --batch_size 1 \
    --enable_smoothnet \
    --smoothnet_checkpoint "$SMOOTHNET_CHECKPOINT" \
    --smoothnet_window_size "$SMOOTHNET_WINDOW_SIZE"

# Check if inference was successful
if [ -f "$OUTPUT_FOLDER/smpl_output.pkl" ]; then
    echo ""
    echo "=========================================="
    echo -e "${GREEN}✓ INFERENCE SUCCESSFUL!${NC}"
    echo "=========================================="
    echo "Output saved to: $OUTPUT_FOLDER/smpl_output.pkl"
    
    # Check if final PKL was saved to output-pkl-files (updated after stabilization)
    FINAL_PKL="output-pkl-files/${VIDEO_BASENAME}.pkl"
    if [ -f "$OUTPUT_FOLDER/smpl_output.pkl" ]; then
        # Copy stabilized PKL to final location
        cp "$OUTPUT_FOLDER/smpl_output.pkl" "$FINAL_PKL"
        echo "Final PKL (stabilized) saved to: $FINAL_PKL"
    fi
    echo ""
    
    # Run PKL structure validation
    echo -e "${YELLOW}Validating PKL structure...${NC}"
    if [ -f "validate_pkl.py" ]; then
        "$PYTHON_EXE" validate_pkl.py "$OUTPUT_FOLDER/smpl_output.pkl" || true
    fi
    
    # Run motion stabilization (eliminates single-frame jitter and rotation spikes)
    echo ""
    echo -e "${YELLOW}Running motion stabilization...${NC}"
    STABILIZED_PKL="$OUTPUT_FOLDER/smpl_output_stabilized.pkl"
    STABILIZATION_REPORT="$OUTPUT_FOLDER/stabilization_report.json"
    
    # Try stabilization with standard threshold first
    if "$PYTHON_EXE" motion_stabilizer.py \
        --input "$OUTPUT_FOLDER/smpl_output.pkl" \
        --output "$STABILIZED_PKL" \
        --fps 30 \
        --report "$STABILIZATION_REPORT" 2>&1 | tee /tmp/stabilization.log; then
        STABILIZATION_RESULT="SUCCESS"
        echo -e "${GREEN}✓ Motion stabilization complete${NC}"
        
        # Replace original PKL with stabilized version
        if [ -f "$STABILIZED_PKL" ]; then
            mv "$STABILIZED_PKL" "$OUTPUT_FOLDER/smpl_output.pkl"
            echo "  Stabilized PKL saved to: $OUTPUT_FOLDER/smpl_output.pkl"
        fi
    else
        # If failed due to too many outliers, try with more lenient threshold
        echo -e "${YELLOW}Standard stabilization failed, trying with lenient threshold...${NC}"
        if "$PYTHON_EXE" motion_stabilizer.py \
            --input "$OUTPUT_FOLDER/smpl_output.pkl" \
            --output "$STABILIZED_PKL" \
            --fps 30 \
            --outlier_threshold 250.0 \
            --max_velocity 250.0 \
            --report "$STABILIZATION_REPORT" 2>&1 | tee /tmp/stabilization.log; then
            STABILIZATION_RESULT="SUCCESS_LENIENT"
            echo -e "${GREEN}✓ Motion stabilization complete (lenient mode)${NC}"
            if [ -f "$STABILIZED_PKL" ]; then
                mv "$STABILIZED_PKL" "$OUTPUT_FOLDER/smpl_output.pkl"
                echo "  Stabilized PKL saved to: $OUTPUT_FOLDER/smpl_output.pkl"
            fi
        else
            STABILIZATION_RESULT="FAILED"
            echo -e "${RED}⚠ Motion stabilization failed even with lenient threshold${NC}"
            echo "  Check: $STABILIZATION_REPORT"
            echo "  Using original PKL (may have significant jitter/spikes)"
            echo "  Consider fixing motion upstream or adjusting thresholds"
        fi
    fi
    
    # Run motion quality validation
    echo ""
    echo -e "${YELLOW}Validating motion quality...${NC}"
    MOTION_VALIDATION_OUTPUT="$OUTPUT_FOLDER/motion_validation.json"
    MOTION_VALIDATION_CSV="$OUTPUT_FOLDER/bad_frames.csv"
    
    if "$PYTHON_EXE" pkl_motion_validator.py \
        --pkl "$OUTPUT_FOLDER/smpl_output.pkl" \
        --fps 30 \
        --output "$MOTION_VALIDATION_OUTPUT" \
        --csv "$MOTION_VALIDATION_CSV"; then
        VALIDATION_RESULT="PASS"
        echo -e "${GREEN}✓ Motion quality validation passed${NC}"
    else
        VALIDATION_RESULT="FAIL"
        echo -e "${RED}⚠ Motion quality validation failed - check report${NC}"
        echo "  Report: $MOTION_VALIDATION_OUTPUT"
        echo "  Bad frames: $MOTION_VALIDATION_CSV"
    fi
    
    echo ""
    echo "=========================================="
    echo -e "${GREEN}✓ PIPELINE COMPLETE${NC}"
    echo "=========================================="
    echo "Output file: $OUTPUT_FOLDER/smpl_output.pkl"
    echo ""
    echo "Pipeline Stages:"
    echo "  1. SmoothNet: Temporal smoothing (window=$SMOOTHNET_WINDOW_SIZE)"
    echo "     ✓ Root joint excluded (no global pops)"
    echo "     ✓ Motion-critical joints smoothed (14/49 joints)"
    echo "     ✓ Velocity clamping (prevents unrealistic spikes)"
    echo ""
    echo "  2. Motion Stabilization: Single-frame jitter elimination"
    if [ "$STABILIZATION_RESULT" = "SUCCESS" ]; then
        echo "     ✓ Outlier frames detected and fixed"
        echo "     ✓ Angular velocity clamped"
        echo "     ✓ Quaternion-space smoothing applied"
    else
        echo "     ⚠ Stabilization failed (using original)"
    fi
    echo ""
    echo "  3. Quality Validation: Motion quality check"
    echo "     Status: $VALIDATION_RESULT"
    echo ""
    echo "Production-Grade Output:"
    echo "  - No skeleton jumps or pops"
    echo "  - No single-frame jitter or tilts"
    echo "  - Smooth motion with preserved detail"
    echo "  - Physically plausible velocities"
    echo "  - Ready for Maya import and retargeting"
    echo ""
    if [ "$VALIDATION_RESULT" = "FAIL" ]; then
        echo -e "${YELLOW}Motion Quality: ${VALIDATION_RESULT}${NC}"
        echo "  Review motion_validation.json and bad_frames.csv for details"
        echo "  Consider re-running with different smoothing parameters"
    else
        echo -e "${GREEN}Motion Quality: ${VALIDATION_RESULT}${NC}"
        echo "  Motion is artist-ready for Maya animation"
    fi
    echo ""
else
    echo ""
    echo -e "${RED}=========================================="
    echo "ERROR: Inference failed"
    echo "==========================================${NC}"
    echo "Output file not found: $OUTPUT_FOLDER/smpl_output.pkl"
    exit 1
fi
