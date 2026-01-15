#!/bin/bash

# Usage: ./run_bfcl_tests.sh <MODEL_PATH> <FOLDER_NAME> <TEST_CATEGORIES>
# Example: ./run_bfcl_tests.sh /path/to/model experiment_v1 "simple_python,parallel"
#
# The model name is hardcoded as "JaisPlus" - only the results folder name is configurable.

set -e

MODEL_PATH="$1"
FOLDER_NAME="$2"
TEST_CATEGORIES="$3"

# Hardcoded model name
MODEL_NAME="JaisPlus"

if [ -z "$MODEL_PATH" ] || [ -z "$FOLDER_NAME" ] || [ -z "$TEST_CATEGORIES" ]; then
    echo "Usage: $0 <MODEL_PATH> <FOLDER_NAME> <TEST_CATEGORIES>"
    echo ""
    echo "Arguments:"
    echo "  MODEL_PATH      - Path to the model checkpoint directory"
    echo "  FOLDER_NAME     - Name of the results folder (unique identifier for this run)"
    echo "  TEST_CATEGORIES - Comma-separated test categories (e.g., 'simple_python,parallel')"
    echo ""
    echo "Example: $0 /path/to/model experiment_v1 'simple_python,parallel'"
    echo ""
    echo "Note: Model name is hardcoded as '$MODEL_NAME'"
    exit 1
fi

RESULTS_BASE="/lustre/scratch/users/abhishek.maiti/BFCL_results"
RESULTS_DIR="${RESULTS_BASE}/${FOLDER_NAME}"

mkdir -p "$RESULTS_DIR"

echo "=========================================="
echo "Running BFCL tests"
echo "Model name: $MODEL_NAME"
echo "Model path: $MODEL_PATH"
echo "Test categories: $TEST_CATEGORIES"
echo "Results folder: $FOLDER_NAME"
echo "Results will be saved to: $RESULTS_DIR"
echo "=========================================="

cd /home/abhishek.maiti/projects/abhishekm2-cerebras/gorilla/berkeley-function-call-leaderboard

# Generate responses
echo "Generating responses..."
bfcl generate \
  --model "$MODEL_NAME" \
  --test-category "$TEST_CATEGORIES" \
  --backend vllm \
  --num-gpus 1 \
  --local-model-path "$MODEL_PATH" \
  --result-dir "$RESULTS_DIR/result"

# Evaluate responses
echo "Evaluating responses..."
bfcl evaluate \
  --model "$MODEL_NAME" \
  --test-category "$TEST_CATEGORIES" \
  --result-dir "$RESULTS_DIR/result" \
  --score-dir "$RESULTS_DIR/score"

echo "=========================================="
echo "Done! Results saved to: $RESULTS_DIR"
echo "=========================================="
