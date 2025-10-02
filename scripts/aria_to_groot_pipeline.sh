#!/bin/bash
################################################################################
# Aria to GR00T Complete Pipeline
#
# This script automates the entire workflow from VRS file to GR00T-ready dataset:
# 1. Convert VRS/MPS to LeRobot format
# 2. Generate GR00T-compatible metadata
# 3. Verify dataset compatibility
# 4. (Optional) Test dataset loading
################################################################################

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
info() { echo -e "${BLUE}ℹ${NC} $1"; }
success() { echo -e "${GREEN}✓${NC} $1"; }
warning() { echo -e "${YELLOW}⚠${NC} $1"; }
error() { echo -e "${RED}✗${NC} $1"; }

################################################################################
# Configuration
################################################################################

# Default values
VRS_FILE=""
OUTPUT_DIR=""
HAPTIC_CSV=""
HAPTIC_OFFSET=0.0
SPLIT_CSV=""
TASK_NAME=""
TASK_DESC=""
SKIP_CONVERSION=false
SKIP_VISUALIZATION=false
TEST_LOADING=false

# Show usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Aria to GR00T Complete Pipeline - Convert VRS data to GR00T-ready dataset

Required Arguments:
    --vrs FILE              Path to VRS file (e.g., glue.vrs)
    --output DIR            Output directory (e.g., output_glue)

Optional Arguments:
    --haptic-csv FILE       Haptic sensor CSV file
    --haptic-offset SEC     Time offset for haptic data (default: 0.0)
    --split-csv FILE        Episode split points CSV
    --task-name NAME        Task name (default: auto-detected from VRS filename)
    --task-desc DESC        Task description

    --skip-conversion       Skip VRS conversion (use existing data)
    --skip-visualization    Skip generating visualization videos
    --test-loading          Test dataset loading with GR00T after conversion

    -h, --help             Show this help message

Examples:
    # Full pipeline with haptic data
    $0 --vrs glue.vrs --output output_glue \\
       --haptic-csv sensor_data/sequence_2.csv \\
       --haptic-offset 1.21 \\
       --split-csv aria_haptic_alignment/glue_segments.csv \\
       --task-name "pick and place glue"

    # Convert existing dataset to GR00T format
    $0 --vrs glue.vrs --output output_glue --skip-conversion

    # With dataset loading test
    $0 --vrs pen.vrs --output output_pen --test-loading

EOF
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --vrs) VRS_FILE="$2"; shift 2 ;;
        --output) OUTPUT_DIR="$2"; shift 2 ;;
        --haptic-csv) HAPTIC_CSV="$2"; shift 2 ;;
        --haptic-offset) HAPTIC_OFFSET="$2"; shift 2 ;;
        --split-csv) SPLIT_CSV="$2"; shift 2 ;;
        --task-name) TASK_NAME="$2"; shift 2 ;;
        --task-desc) TASK_DESC="$2"; shift 2 ;;
        --skip-conversion) SKIP_CONVERSION=true; shift ;;
        --skip-visualization) SKIP_VISUALIZATION=true; shift ;;
        --test-loading) TEST_LOADING=true; shift ;;
        -h|--help) usage ;;
        *) error "Unknown option: $1"; usage ;;
    esac
done

# Validate required arguments
if [[ -z "$VRS_FILE" ]] || [[ -z "$OUTPUT_DIR" ]]; then
    error "Missing required arguments"
    usage
fi

# Auto-detect task name if not provided
if [[ -z "$TASK_NAME" ]]; then
    BASENAME=$(basename "$VRS_FILE" .vrs)
    TASK_NAME="pick and place ${BASENAME}"
    info "Auto-detected task name: $TASK_NAME"
fi

# Auto-detect task description if not provided
if [[ -z "$TASK_DESC" ]]; then
    TASK_DESC="Manipulation task with Aria recording: ${TASK_NAME}"
fi

################################################################################
# Pipeline Steps
################################################################################

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Aria to GR00T Pipeline"
echo "════════════════════════════════════════════════════════════════"
echo ""
info "VRS File: $VRS_FILE"
info "Output Directory: $OUTPUT_DIR"
info "Task: $TASK_NAME"
echo ""

# Activate conda environment
info "Step 0: Activating conda environment..."
source /home/echo/anaconda3/etc/profile.d/conda.sh
conda activate lerobot
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/aria_api"
success "Environment activated"
echo ""

# Step 1: Convert VRS to LeRobot format
if [[ "$SKIP_CONVERSION" == false ]]; then
    info "Step 1: Converting VRS/MPS to LeRobot format with GR00T compatibility..."

    CONVERT_CMD="python -m aria2lerobot \"$VRS_FILE\" \"$OUTPUT_DIR\" \
        --task-name \"$TASK_NAME\" \
        --task-description \"$TASK_DESC\" \
        --groot-format \
        --overwrite"

    # Add optional arguments
    [[ -n "$HAPTIC_CSV" ]] && CONVERT_CMD="$CONVERT_CMD --haptic-csv \"$HAPTIC_CSV\""
    [[ "$HAPTIC_OFFSET" != "0.0" ]] && CONVERT_CMD="$CONVERT_CMD --haptic-offset $HAPTIC_OFFSET"
    [[ -n "$SPLIT_CSV" ]] && CONVERT_CMD="$CONVERT_CMD --split-csv \"$SPLIT_CSV\""

    if [[ "$SKIP_VISUALIZATION" == false ]]; then
        CONVERT_CMD="$CONVERT_CMD --create-visualization --rotate-visualization"
    fi

    echo "  Command: $CONVERT_CMD"
    eval $CONVERT_CMD

    if [[ $? -eq 0 ]]; then
        success "VRS conversion complete"
    else
        error "VRS conversion failed"
        exit 1
    fi
else
    warning "Skipping VRS conversion (using existing data)"
fi
echo ""

# Step 2: Verify GR00T format
info "Step 2: Verifying GR00T format compatibility..."
python3 aria2lerobot/tools/convert_to_groot_format.py "$OUTPUT_DIR"

if [[ $? -eq 0 ]]; then
    success "GR00T format verified"
else
    error "GR00T format verification failed"
    exit 1
fi
echo ""

# Step 3: Test dataset loading (optional)
if [[ "$TEST_LOADING" == true ]]; then
    info "Step 3: Testing dataset loading with GR00T..."
    python3 aria2lerobot/tools/load_groot_dataset.py "$OUTPUT_DIR" --sample-index 10

    if [[ $? -eq 0 ]]; then
        success "Dataset loading test passed"
    else
        error "Dataset loading test failed"
        exit 1
    fi
else
    info "Step 3: Skipping dataset loading test (use --test-loading to enable)"
fi
echo ""

# Final summary
echo "════════════════════════════════════════════════════════════════"
echo "  Pipeline Complete! 🎉"
echo "════════════════════════════════════════════════════════════════"
echo ""
success "Dataset ready: $OUTPUT_DIR"
echo ""
echo "Dataset Statistics:"
echo "  • Total episodes: $(cat \"$OUTPUT_DIR/meta/episodes.jsonl\" | wc -l)"
echo "  • Data files: $(find \"$OUTPUT_DIR/data\" -name '*.parquet' | wc -l)"
echo "  • Video files: $(find \"$OUTPUT_DIR/videos\" -name '*.mp4' | wc -l)"
[[ "$SKIP_VISUALIZATION" == false ]] && echo "  • Visualizations: $(find \"$OUTPUT_DIR\" -maxdepth 1 -name 'visualization_*.mp4' | wc -l)"
echo ""
echo "Next Steps:"
echo "  1. Load dataset: python3 aria2lerobot/tools/load_groot_dataset.py $OUTPUT_DIR"
echo "  2. Train GR00T: (use your training script with embodiment_tag='new_embodiment')"
echo ""
