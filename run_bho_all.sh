#!/bin/bash
# Run BHO for all three models in parallel

echo "========================================="
echo "BLACK HOLE OPTIMIZATION - ALL MODELS"
echo "========================================="
echo "Starting optimization for MFCC, MelSpec, and Chroma"
echo "This will take approximately 3-4 hours"
echo "========================================="
echo ""

# Create logs directory
mkdir -p logs

# Get timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Start MFCC optimization
echo "🚀 Starting MFCC optimization..."
nohup python black_hole_optimizer_fixed.py \
  --feature mfcc \
  --population 12 \
  --iterations 15 \
  --eval_epochs 20 \
  --device mps \
  > logs/bho_mfcc_${TIMESTAMP}.log 2>&1 &
PID_MFCC=$!
echo "   MFCC PID: $PID_MFCC"

# Start MelSpec optimization
echo "🚀 Starting MelSpec optimization..."
nohup python black_hole_optimizer_fixed.py \
  --feature melspec \
  --population 12 \
  --iterations 15 \
  --eval_epochs 20 \
  --device mps \
  > logs/bho_melspec_${TIMESTAMP}.log 2>&1 &
PID_MELSPEC=$!
echo "   MelSpec PID: $PID_MELSPEC"

# Start Chroma optimization
echo "🚀 Starting Chroma optimization..."
nohup python black_hole_optimizer_fixed.py \
  --feature chroma \
  --population 12 \
  --iterations 15 \
  --eval_epochs 20 \
  --device mps \
  > logs/bho_chroma_${TIMESTAMP}.log 2>&1 &
PID_CHROMA=$!
echo "   Chroma PID: $PID_CHROMA"

echo ""
echo "========================================="
echo "✅ All optimizations started!"
echo "========================================="
echo "Process IDs:"
echo "  MFCC:    $PID_MFCC"
echo "  MelSpec: $PID_MELSPEC"
echo "  Chroma:  $PID_CHROMA"
echo ""
echo "Monitor progress with:"
echo "  tail -f logs/bho_mfcc_${TIMESTAMP}.log"
echo "  tail -f logs/bho_melspec_${TIMESTAMP}.log"
echo "  tail -f logs/bho_chroma_${TIMESTAMP}.log"
echo ""
echo "Check status with:"
echo "  ps aux | grep black_hole_optimizer"
echo ""
echo "Results will be in:"
echo "  models/bho_results/mfcc_*/"
echo "  models/bho_results/melspec_*/"
echo "  models/bho_results/chroma_*/"
echo "========================================="