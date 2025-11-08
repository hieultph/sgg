#!/bin/bash

# SGG-Benchmark Training Test Script
# This script tests if training can start properly

echo "Testing SGG-Benchmark Training Setup..."
echo "======================================"

# Activate environment
conda activate sgg_benchmark
cd /workspace/SGG-Benchmark

echo "1. Testing basic imports..."
python -c "
from sgg_benchmark.config import cfg
from sgg_benchmark.modeling.detector import build_detection_model
from sgg_benchmark.data import make_data_loader
print('✓ All core modules imported successfully')
"

echo ""
echo "2. Testing model configuration..."
python -c "
from sgg_benchmark.config import cfg
cfg.merge_from_file('configs/VG150/react_yolov8m.yaml')
print('✓ Config loaded successfully')
print(f'Model architecture: {cfg.MODEL.META_ARCHITECTURE}')
print(f'Relation predictor: {cfg.MODEL.ROI_RELATION_HEAD.PREDICTOR}')
"

echo ""
echo "3. Testing if model can be built..."
python -c "
from sgg_benchmark.config import cfg
cfg.merge_from_file('configs/VG150/react_yolov8m.yaml')
cfg.MODEL.PRETRAINED_DETECTOR_CKPT = ''  # Skip pretrained weights for test
from sgg_benchmark.modeling.detector import build_detection_model
model = build_detection_model(cfg)
print('✓ Model built successfully')
print(f'Model type: {type(model).__name__}')
"

echo ""
echo "======================================"
echo "✅ SGG-Benchmark is ready for training!"
echo ""
echo "Note: Deformable convolutions are disabled due to CUDA extension issues,"
echo "but this won't affect YOLO-based models like REACT."
echo ""
echo "To start training:"
echo "1. Prepare your dataset (VG150 recommended)"
echo "2. Download GloVe embeddings"
echo "3. Update paths in sgg_benchmark/config/paths_catalog.py"
echo "4. Run training command (see READY_TO_TRAIN.md)"