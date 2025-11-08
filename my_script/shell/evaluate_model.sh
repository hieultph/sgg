#!/bin/bash

# Evaluation Script for SGG Models

# Set your paths
GLOVE_DIR="/path/to/glove"
MODEL_DIR="./checkpoints/my_custom_model_sgdet"
CONFIG_FILE="$MODEL_DIR/config.yml"  # Auto-saved during training

echo "Evaluating SGG model..."

# Test SGDet model
echo "Testing SGDet performance..."
CUDA_VISIBLE_DEVICES=0 python tools/relation_test_net.py \
    --config-file $CONFIG_FILE \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    MODEL.ROI_RELATION_HEAD.PREDICTOR REACTPredictor \
    TEST.IMS_PER_BATCH 1 \
    DTYPE "float16" \
    GLOVE_DIR $GLOVE_DIR \
    MODEL.PRETRAINED_DETECTOR_CKPT $MODEL_DIR \
    OUTPUT_DIR $MODEL_DIR

# Test PredCls performance (if you have a PredCls model)
echo "Testing PredCls performance..."
PREDCLS_MODEL_DIR="./checkpoints/my_custom_model_predcls"
CUDA_VISIBLE_DEVICES=0 python tools/relation_test_net.py \
    --config-file "$PREDCLS_MODEL_DIR/config.yml" \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX True \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL True \
    MODEL.ROI_RELATION_HEAD.PREDICTOR REACTPredictor \
    TEST.IMS_PER_BATCH 1 \
    DTYPE "float16" \
    GLOVE_DIR $GLOVE_DIR \
    MODEL.PRETRAINED_DETECTOR_CKPT $PREDCLS_MODEL_DIR \
    OUTPUT_DIR $PREDCLS_MODEL_DIR

echo "Evaluation completed! Check the output directory for results."