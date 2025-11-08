#!/bin/bash

# Custom SGG Training Script
# Modify paths and parameters according to your setup

# Set your paths
GLOVE_DIR="/path/to/glove"  # Download GloVe embeddings
OUTPUT_DIR="./checkpoints/my_custom_model"
CONFIG_FILE="configs/VG150/react_yolov8m.yaml"

# Training configurations
GPUS="0,1"  # Specify GPU IDs
BATCH_SIZE=8
TEST_BATCH_SIZE=1
MAX_EPOCHS=20

echo "Starting SGG training..."

# 1. PREDICATE CLASSIFICATION (PredCls) - Easiest, uses GT boxes and labels
echo "Training PredCls model..."
CUDA_VISIBLE_DEVICES=$GPUS python -m torch.distributed.launch \
    --master_port 10001 --nproc_per_node=2 \
    tools/relation_train_net.py \
    --task predcls \
    --save-best \
    --config-file $CONFIG_FILE \
    MODEL.ROI_RELATION_HEAD.PREDICTOR REACTPredictor \
    SOLVER.IMS_PER_BATCH $BATCH_SIZE \
    TEST.IMS_PER_BATCH $TEST_BATCH_SIZE \
    DTYPE "float16" \
    SOLVER.MAX_EPOCH $MAX_EPOCHS \
    GLOVE_DIR $GLOVE_DIR \
    OUTPUT_DIR "${OUTPUT_DIR}_predcls"

# 2. SCENE GRAPH DETECTION (SGDet) - Most challenging, detects everything from scratch
echo "Training SGDet model..."
CUDA_VISIBLE_DEVICES=$GPUS python -m torch.distributed.launch \
    --master_port 10002 --nproc_per_node=2 \
    tools/relation_train_net.py \
    --task sgdet \
    --save-best \
    --use-wandb \
    --config-file $CONFIG_FILE \
    MODEL.ROI_RELATION_HEAD.PREDICTOR REACTPredictor \
    SOLVER.IMS_PER_BATCH $BATCH_SIZE \
    TEST.IMS_PER_BATCH $TEST_BATCH_SIZE \
    DTYPE "float16" \
    SOLVER.MAX_EPOCH $MAX_EPOCHS \
    GLOVE_DIR $GLOVE_DIR \
    OUTPUT_DIR "${OUTPUT_DIR}_sgdet"

echo "Training completed!"