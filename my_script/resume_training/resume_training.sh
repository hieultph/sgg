#!/bin/bash
# Script to resume training from checkpoint
# Usage: bash resume_training.sh

set -e  # Exit on error

echo "======================================================================"
echo "🚀 Resume Training Script for PSG REACT Model"
echo "======================================================================"

# Configuration
CHECKPOINT_SRC="checkpoints/react_PSG/best_model_epoch_11.pth"
OUTPUT_DIR="output/react_PSG_resume"
CONFIG_FILE="configs/PSG/react_yolov8m_resume.yaml"
GPU_ID=0

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_SRC" ]; then
    echo "❌ Error: Checkpoint not found at $CHECKPOINT_SRC"
    exit 1
fi

echo "✓ Checkpoint found: $CHECKPOINT_SRC"

# Create output directory
mkdir -p "$OUTPUT_DIR"
echo "✓ Created output directory: $OUTPUT_DIR"

# Copy checkpoint to output directory with standard name
echo "📦 Copying checkpoint..."
cp "$CHECKPOINT_SRC" "$OUTPUT_DIR/model_final.pth"
echo "✓ Checkpoint copied to: $OUTPUT_DIR/model_final.pth"

# Check if conda environment is activated
if [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo "⚠️  Warning: No conda environment detected"
    echo "   Please run: conda activate sgg_benchmark"
    exit 1
fi

echo "✓ Conda environment: $CONDA_DEFAULT_ENV"

# Check if config exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Config file not found at $CONFIG_FILE"
    exit 1
fi

echo "✓ Config file: $CONFIG_FILE"

# Create logs directory
mkdir -p logs
LOGFILE="logs/resume_training_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "======================================================================"
echo "📊 Training Configuration"
echo "======================================================================"
echo "  Checkpoint: $CHECKPOINT_SRC"
echo "  Output Dir: $OUTPUT_DIR"
echo "  Config:     $CONFIG_FILE"
echo "  GPU:        $GPU_ID"
echo "  Log File:   $LOGFILE"
echo "======================================================================"
echo ""

# Ask for confirmation
read -p "Start training? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Training cancelled"
    exit 0
fi

echo ""
echo "🎯 Starting training..."
echo "   Press Ctrl+C to stop"
echo ""

# Start training
CUDA_VISIBLE_DEVICES=$GPU_ID python tools/relation_train_net.py \
    --config-file "$CONFIG_FILE" \
    2>&1 | tee "$LOGFILE"

echo ""
echo "======================================================================"
echo "✅ Training completed!"
echo "   Log saved to: $LOGFILE"
echo "   Checkpoints saved to: $OUTPUT_DIR"
echo "======================================================================"
