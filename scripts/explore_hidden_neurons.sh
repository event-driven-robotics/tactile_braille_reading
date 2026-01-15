#!/bin/bash

###############################################################################
# Explore RSNN Performance Across Different Hidden Layer Sizes
#
# This script systematically trains the braille-reading RSNN with varying
# numbers of hidden neurons and collects performance metrics for analysis.
#
# Usage:
#   ./scripts/explore_hidden_neurons.sh [OPTIONS]
#
# Options:
#   --neurons "10 20 50 100"    Neuron counts to test (default: 10 20 50 100)
#   --epochs 50                 Training epochs per run (default: 50)
#   --repetitions 3             Training repetitions per config (default: 3)
#   --letters "A B"             Letters to classify (default: A B)
#   --learning-rate 0.0001      Learning rate (default: 0.0001)
#   --batch-size 128            Batch size (default: 128)
#   --use-eprop                 Use e-prop instead of BPTT (default: false)
#   --use-seed                  Use fixed seed for reproducibility
#   --help                      Show this help message
#
# Examples:
#   # Quick test with small networks
#   ./scripts/explore_hidden_neurons.sh --neurons "10 20 30" --epochs 10 --repetitions 1
#
#   # Comprehensive exploration
#   ./scripts/explore_hidden_neurons.sh --neurons "25 50 75 100 150" --epochs 50 --repetitions 5
#
#   # E-prop with seed for reproducibility
#   ./scripts/explore_hidden_neurons.sh --use-eprop --use-seed
#
# Output:
#   - Results saved to: ./results/YYYYMMDD_HHMM_exploration/
#   - Models saved to: ./model/YYYYMMDD_HHMM_exploration/
#   - Figures saved to: ./figures/YYYYMMDD_HHMM_exploration/
#   - Summary file: ./results/exploration_summary_YYYYMMDD_HHMM.txt
#
# Author: Generated for tactile braille reading project
###############################################################################

set -e  # Exit on error

# Default parameters
NEURONS=(5 10 20 50)
EPOCHS=50
REPETITIONS=5
LETTERS=("A" "B")
LEARNING_RATE=0.00005
BATCH_SIZE=128
USE_EPROP=false
USE_SEED=false
# VENV_PYTHON="/home/smullercleve/.virtualenvs/pytorch/bin/python"
VENV_PYTHON="/home/smullercleve-iit.local/.virtualenvs/pytorch/bin/python"  # WS
SCRIPT_PATH="scripts/braille_reading_rsnn_mod_eprop_reduce_label.py"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --neurons)
            NEURONS=($2)
            shift 2
            ;;
        --epochs)
            EPOCHS=$2
            shift 2
            ;;
        --repetitions)
            REPETITIONS=$2
            shift 2
            ;;
        --letters)
            LETTERS=($2)
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE=$2
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE=$2
            shift 2
            ;;
        --use-eprop)
            USE_EPROP=true
            shift
            ;;
        --use-seed)
            USE_SEED=true
            shift
            ;;
        --help)
            grep "^#" "$0" | grep -v "#!/bin/bash" | sed 's/^# //'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Verify venv exists
if [ ! -f "$VENV_PYTHON" ]; then
    echo -e "${RED}Error: Virtual environment not found at $VENV_PYTHON${NC}"
    exit 1
fi

# Create exploration-specific directories
TIMESTAMP=$(date +%Y%m%d_%H%M)
EXPLORATION_DIR="${TIMESTAMP}_exploration"
RESULTS_DIR="./results/${EXPLORATION_DIR}"
MODELS_DIR="./model/${EXPLORATION_DIR}"
FIGURES_DIR="./figures/${EXPLORATION_DIR}"

mkdir -p "$RESULTS_DIR" "$MODELS_DIR" "$FIGURES_DIR"

# Create summary file
SUMMARY_FILE="./results/exploration_summary_${TIMESTAMP}.txt"

# Print header
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}RSNN Hidden Neuron Exploration${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""
echo "Configuration:"
echo "  Neurons to test:     ${NEURONS[@]}"
echo "  Epochs per run:      $EPOCHS"
echo "  Repetitions:         $REPETITIONS"
echo "  Letters:             ${LETTERS[@]}"
echo "  Learning rate:       $LEARNING_RATE"
echo "  Batch size:          $BATCH_SIZE"
echo "  Algorithm:           $([ "$USE_EPROP" = true ] && echo "e-prop" || echo "BPTT")"
echo "  Seed:                $([ "$USE_SEED" = true ] && echo "Fixed (42)" || echo "Random")"
echo ""
echo "Output directories:"
echo "  Results: $RESULTS_DIR"
echo "  Models:  $MODELS_DIR"
echo "  Figures: $FIGURES_DIR"
echo ""

# Initialize summary with header
{
    echo "RSNN Exploration Summary"
    echo "================================"
    echo "Timestamp: $(date)"
    echo "Configuration:"
    echo "  Epochs: $EPOCHS"
    echo "  Repetitions: $REPETITIONS"
    echo "  Learning rate: $LEARNING_RATE"
    echo "  Batch size: $BATCH_SIZE"
    echo "  Algorithm: $([ "$USE_EPROP" = true ] && echo "e-prop" || echo "BPTT")"
    echo ""
    echo "Results:"
    echo "--------"
} > "$SUMMARY_FILE"

# Track total start time
TOTAL_START=$(date +%s)
TOTAL_CONFIGS=${#NEURONS[@]}
CURRENT_CONFIG=0

# Loop through each neuron count
for NB_HIDDEN in "${NEURONS[@]}"; do
    CURRENT_CONFIG=$((CURRENT_CONFIG + 1))
    
    echo -e "${GREEN}[${CURRENT_CONFIG}/${TOTAL_CONFIGS}] Testing with ${NB_HIDDEN} hidden neurons${NC}"
    echo ""
    
    CONFIG_START=$(date +%s)
    
    # Build command
    CMD=(
        "$VENV_PYTHON"
        "$SCRIPT_PATH"
        "--epochs" "$EPOCHS"
        "--nb_hidden" "$NB_HIDDEN"
        "--batch_size" "$BATCH_SIZE"
        "--learning_rate" "$LEARNING_RATE"
        "--repetitions" "$REPETITIONS"
        "--letters" "${LETTERS[@]}"
        "--fig_path" "$FIGURES_DIR"
        "--model_path" "$MODELS_DIR"
        "--results_path" "$RESULTS_DIR"
    )
    
    # Add optional flags
    [ "$USE_EPROP" = true ] && CMD+=("--use_eprop")
    [ "$USE_SEED" = true ] && CMD+=("--use_seed")
    
    # Run training
    if "${CMD[@]}"; then
        CONFIG_END=$(date +%s)
        CONFIG_DURATION=$((CONFIG_END - CONFIG_START))
        
        echo -e "${GREEN}✓ Completed in $((CONFIG_DURATION / 60))m $((CONFIG_DURATION % 60))s${NC}"
        
        # Try to find the results file (may be in a timestamped subdirectory)
        LETTERS_STR=$(printf "_%s" "${LETTERS[@]}")
        LETTERS_STR=${LETTERS_STR:1}
        RESULTS_FILE=$(find "$RESULTS_DIR" -name "braille_reading_rsnn_${NB_HIDDEN}_neurons_${LETTERS_STR}.npz" -type f 2>/dev/null | head -n 1)
        if [ -n "$RESULTS_FILE" ]; then
            echo "  Results: $RESULTS_FILE"
            echo "" >> "$SUMMARY_FILE"
            echo "✓ ${NB_HIDDEN} neurons - Completed in $((CONFIG_DURATION / 60))m $((CONFIG_DURATION % 60))s" >> "$SUMMARY_FILE"
        else
            echo "  Warning: Results file not found"
        fi
    else
        echo -e "${YELLOW}✗ Failed${NC}"
        echo "" >> "$SUMMARY_FILE"
        echo "✗ ${NB_HIDDEN} neurons - FAILED" >> "$SUMMARY_FILE"
    fi
    
    echo ""
done

# Calculate total time
TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))

# Print summary
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Exploration Complete!${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}"
echo ""
echo "Total duration: $((TOTAL_DURATION / 60))m $((TOTAL_DURATION % 60))s"
echo "Average per config: $((TOTAL_DURATION / TOTAL_CONFIGS))s"
echo ""
echo "Summary saved to: $SUMMARY_FILE"
echo "Results available in: $RESULTS_DIR"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "  1. Analyze results: python scripts/analyze_exploration_results.py --exploration-dir $RESULTS_DIR"
echo "  2. View plots: ls $FIGURES_DIR/*/"
echo "  3. Check summary: cat $SUMMARY_FILE"
echo ""
