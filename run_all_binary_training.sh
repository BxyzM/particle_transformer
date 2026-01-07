#!/bin/bash

# Run all binary classification training tasks sequentially
# Usage: ./run_all_binary_training.sh [model] [feature_type]
# Example: ./run_all_binary_training.sh ParT kin

set -e  # Exit on error

MODEL=${1:-ParT}
FEATURE_TYPE=${2:-kin}

echo "=========================================="
echo "Starting all binary classification training"
echo "Model: ${MODEL}"
echo "Feature Type: ${FEATURE_TYPE}"
echo "=========================================="



echo ""
echo "=========================================="
echo "Task 2/3: WZ_vs_QCD"
echo "=========================================="
./train_JetClass_Binary.sh ${MODEL} ${FEATURE_TYPE} WZ_vs_QCD 
echo "✓ WZ_vs_QCD completed"


echo ""
echo "=========================================="
echo "Task 3/3: HToCC_vs_QCD"
echo "=========================================="
./train_JetClass_Binary.sh ${MODEL} ${FEATURE_TYPE} HToCC_vs_QCD 
echo "✓ HToCC_vs_QCD completed"

echo ""
echo "=========================================="
echo "All binary classification training completed!"
echo "=========================================="
