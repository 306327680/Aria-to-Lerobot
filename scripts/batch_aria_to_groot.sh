#!/bin/bash
################################################################################
# Batch Aria to GR00T Pipeline
#
# This script processes multiple VRS files in batch using the aria_to_groot_pipeline.sh
# Based on the configurations from run_conversion.sh
################################################################################

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Helper functions
info() { echo -e "${BLUE}ℹ${NC} $1"; }
success() { echo -e "${GREEN}✓${NC} $1"; }
warning() { echo -e "${YELLOW}⚠${NC} $1"; }
error() { echo -e "${RED}✗${NC} $1"; }
header() { echo -e "${CYAN}$1${NC}"; }

################################################################################
# Configuration Array
################################################################################

# Define all conversion tasks
# Format: "VRS_FILE|OUTPUT_DIR|HAPTIC_CSV|HAPTIC_OFFSET|SPLIT_CSV|TASK_NAME|TASK_DESC"

declare -a TASKS=(
    "glue.vrs|output_glue|sensor_data/sequence_2_20250927_203243.csv|1.21|aria_haptic_alignment/glue_alignment_segments.csv|pick and place glue|Assembly task using glue application"
    "pen.vrs|output_pen|sensor_data/sequence_3_20250927_203920.csv|1.13|aria_haptic_alignment/pen_alignment_segments.csv|pick and place pen|Assembly task using pen placement"
    "screw.vrs|output_screw|sensor_data/sequence_1_20250927_202614.csv|0.91|aria_haptic_alignment/screw_alignment_segments.csv|pick and place screw|Assembly task using screw fastening"
)

################################################################################
# Options
################################################################################

SKIP_CONVERSION=false
SKIP_VISUALIZATION=false
TEST_LOADING=false
PARALLEL=false
SELECTED_TASKS=()

# Parse command line arguments
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Batch Aria to GR00T Pipeline - Process multiple VRS files

Options:
    --skip-conversion       Skip VRS conversion (use existing data)
    --skip-visualization    Skip generating visualization videos
    --test-loading          Test dataset loading with GR00T after conversion
    --parallel              Run conversions in parallel (experimental)

    --only TASK_NAME        Only process specific task(s) (can be used multiple times)
                           Valid names: glue, pen, screw

    -h, --help             Show this help message

Examples:
    # Process all tasks
    $0

    # Process all with loading test
    $0 --test-loading

    # Only process glue and pen
    $0 --only glue --only pen

    # Skip conversion, only update GR00T metadata
    $0 --skip-conversion

    # Process in parallel (faster but harder to debug)
    $0 --parallel

EOF
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-conversion) SKIP_CONVERSION=true; shift ;;
        --skip-visualization) SKIP_VISUALIZATION=true; shift ;;
        --test-loading) TEST_LOADING=true; shift ;;
        --parallel) PARALLEL=true; shift ;;
        --only) SELECTED_TASKS+=("$2"); shift 2 ;;
        -h|--help) usage ;;
        *) error "Unknown option: $1"; usage ;;
    esac
done

################################################################################
# Main Processing
################################################################################

echo ""
header "════════════════════════════════════════════════════════════════"
header "  Batch Aria to GR00T Pipeline"
header "════════════════════════════════════════════════════════════════"
echo ""

# Filter tasks if --only was used
FILTERED_TASKS=()
if [[ ${#SELECTED_TASKS[@]} -gt 0 ]]; then
    info "Filtering tasks: ${SELECTED_TASKS[*]}"
    for task in "${TASKS[@]}"; do
        TASK_NAME=$(echo "$task" | cut -d'|' -f6)
        for selected in "${SELECTED_TASKS[@]}"; do
            if [[ "$TASK_NAME" == *"$selected"* ]]; then
                FILTERED_TASKS+=("$task")
                break
            fi
        done
    done
else
    FILTERED_TASKS=("${TASKS[@]}")
fi

if [[ ${#FILTERED_TASKS[@]} -eq 0 ]]; then
    error "No tasks to process!"
    exit 1
fi

info "Total tasks to process: ${#FILTERED_TASKS[@]}"
echo ""

# Function to process a single task
process_task() {
    local task_config="$1"
    local task_index="$2"
    local total_tasks="$3"

    # Parse task configuration
    IFS='|' read -r VRS_FILE OUTPUT_DIR HAPTIC_CSV HAPTIC_OFFSET SPLIT_CSV TASK_NAME TASK_DESC <<< "$task_config"

    header "────────────────────────────────────────────────────────────────"
    header "  Task $task_index/$total_tasks: $TASK_NAME"
    header "────────────────────────────────────────────────────────────────"
    info "VRS: $VRS_FILE → Output: $OUTPUT_DIR"
    echo ""

    # Build command
    CMD="bash aria2lerobot/scripts/aria_to_groot_pipeline.sh"
    CMD="$CMD --vrs \"$VRS_FILE\""
    CMD="$CMD --output \"$OUTPUT_DIR\""
    CMD="$CMD --task-name \"$TASK_NAME\""
    CMD="$CMD --task-desc \"$TASK_DESC\""

    # Add optional arguments
    [[ -n "$HAPTIC_CSV" ]] && CMD="$CMD --haptic-csv \"$HAPTIC_CSV\""
    [[ -n "$HAPTIC_OFFSET" && "$HAPTIC_OFFSET" != "0.0" ]] && CMD="$CMD --haptic-offset $HAPTIC_OFFSET"
    [[ -n "$SPLIT_CSV" ]] && CMD="$CMD --split-csv \"$SPLIT_CSV\""

    [[ "$SKIP_CONVERSION" == true ]] && CMD="$CMD --skip-conversion"
    [[ "$SKIP_VISUALIZATION" == true ]] && CMD="$CMD --skip-visualization"
    [[ "$TEST_LOADING" == true ]] && CMD="$CMD --test-loading"

    # Execute
    echo "Command: $CMD"
    echo ""

    if eval $CMD; then
        success "Task $task_index/$total_tasks completed: $TASK_NAME"
    else
        error "Task $task_index/$total_tasks failed: $TASK_NAME"
        return 1
    fi

    echo ""
}

# Process tasks
TOTAL_TASKS=${#FILTERED_TASKS[@]}
START_TIME=$(date +%s)

if [[ "$PARALLEL" == true ]]; then
    warning "Parallel mode enabled (experimental)"
    info "Running $TOTAL_TASKS tasks in parallel..."
    echo ""

    PIDS=()
    for i in "${!FILTERED_TASKS[@]}"; do
        process_task "${FILTERED_TASKS[$i]}" $((i+1)) $TOTAL_TASKS &
        PIDS+=($!)
    done

    # Wait for all background processes
    FAILED=0
    for pid in "${PIDS[@]}"; do
        if ! wait $pid; then
            FAILED=$((FAILED + 1))
        fi
    done

    if [[ $FAILED -gt 0 ]]; then
        error "$FAILED tasks failed"
        exit 1
    fi
else
    # Sequential processing
    for i in "${!FILTERED_TASKS[@]}"; do
        process_task "${FILTERED_TASKS[$i]}" $((i+1)) $TOTAL_TASKS || exit 1
    done
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

################################################################################
# Summary
################################################################################

header "════════════════════════════════════════════════════════════════"
header "  Batch Processing Complete! 🎉"
header "════════════════════════════════════════════════════════════════"
echo ""
success "Processed $TOTAL_TASKS tasks in ${ELAPSED}s"
echo ""

echo "Summary:"
for task_config in "${FILTERED_TASKS[@]}"; do
    IFS='|' read -r VRS_FILE OUTPUT_DIR HAPTIC_CSV HAPTIC_OFFSET SPLIT_CSV TASK_NAME TASK_DESC <<< "$task_config"

    if [[ -d "$OUTPUT_DIR" ]]; then
        EPISODES=$(cat "$OUTPUT_DIR/meta/episodes.jsonl" 2>/dev/null | wc -l || echo "0")
        DATA_FILES=$(find "$OUTPUT_DIR/data" -name '*.parquet' 2>/dev/null | wc -l || echo "0")
        VIDEO_FILES=$(find "$OUTPUT_DIR/videos" -name '*.mp4' 2>/dev/null | wc -l || echo "0")

        echo "  ✓ $TASK_NAME ($OUTPUT_DIR):"
        echo "    - Episodes: $EPISODES"
        echo "    - Data files: $DATA_FILES"
        echo "    - Video files: $VIDEO_FILES"
    else
        echo "  ✗ $TASK_NAME ($OUTPUT_DIR): Not found"
    fi
done

echo ""
echo "Next Steps:"
echo "  1. Load any dataset:"
for task_config in "${FILTERED_TASKS[@]}"; do
    OUTPUT_DIR=$(echo "$task_config" | cut -d'|' -f2)
    echo "     python3 aria2lerobot/tools/load_groot_dataset.py $OUTPUT_DIR --sample-index 100"
done
echo ""
echo "  2. Start training with your preferred configuration:"
echo "     python3 aria2lerobot/tools/train_example_pinch_grasp.py"
echo ""
