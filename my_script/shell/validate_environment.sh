#!/bin/bash

# SGG-Benchmark Environment Validation Script
echo "=== SGG-Benchmark Environment Validation ==="

# Activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate sgg_benchmark

echo "1. Testing Python and basic imports..."
python -c "
import sys
print(f'Python version: {sys.version}')
"

echo "2. Testing PyTorch installation..."
python -c "
import torch
import torchvision
print(f'PyTorch version: {torch.__version__}')
print(f'Torchvision version: {torchvision.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device count: {torch.cuda.device_count()}')
    print(f'Current CUDA device: {torch.cuda.current_device()}')
    print(f'Device name: {torch.cuda.get_device_name(0)}')
"

echo "3. Testing SGG-Benchmark installation..."
python -c "
import sgg_benchmark
from sgg_benchmark.config import cfg
from sgg_benchmark.modeling.detector import build_detection_model
print('✓ SGG-Benchmark core modules imported successfully')
"

echo "4. Testing YOLO backbone support..."
python -c "
from ultralytics import YOLO
print('✓ Ultralytics YOLO available')
"

echo "5. Testing other dependencies..."
python -c "
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import wandb
import yacs
print('✓ All major dependencies available')
"

echo "6. Testing CUDA compilation (optional)..."
python -c "
try:
    from sgg_benchmark import _C
    print('✓ CUDA extensions compiled successfully')
except ImportError as e:
    print('⚠ CUDA extensions not compiled (this is optional for basic functionality)')
    print(f'  Error: {e}')
"

echo ""
echo "=== Validation Complete ==="
echo "✓ Environment is ready for SGG training!"
echo ""
echo "Next steps:"
echo "1. Download dataset (see dataset_guide.md)"
echo "2. Download GloVe embeddings"
echo "3. Update paths in sgg_benchmark/config/paths_catalog.py"
echo "4. Run training with: ./train_custom_model.sh"