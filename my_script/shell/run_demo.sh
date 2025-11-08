#!/bin/bash

# Demo script for SGG inference

MODEL_DIR="./checkpoints/my_custom_model_sgdet"
CONFIG_FILE="$MODEL_DIR/config.yml"
WEIGHTS_FILE="$MODEL_DIR/model_best.pth"
CUSTOM_IMAGES_DIR="./demo/custom_images"

echo "Running SGG demo..."

# Method 1: Webcam demo (real-time)
echo "Starting webcam demo..."
python demo/webcam_demo.py \
    --config $CONFIG_FILE \
    --weights $WEIGHTS_FILE \
    --rel_conf 0.1 \
    --box_conf 0.5 \
    --tracking  # Optional: requires boxmot

# Method 2: Process custom images
echo "Processing custom images..."
mkdir -p $CUSTOM_IMAGES_DIR

# Test on custom images using the test script
CUDA_VISIBLE_DEVICES=0 python tools/relation_test_net.py \
    --config-file $CONFIG_FILE \
    MODEL.ROI_RELATION_HEAD.USE_GT_BOX False \
    MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL False \
    TEST.IMS_PER_BATCH 1 \
    DTYPE "float16" \
    MODEL.PRETRAINED_DETECTOR_CKPT $WEIGHTS_FILE \
    OUTPUT_DIR $MODEL_DIR \
    TEST.CUSTUM_EVAL True \
    TEST.CUSTUM_PATH $CUSTOM_IMAGES_DIR \
    DETECTED_SGG_DIR "${MODEL_DIR}/custom_results"

echo "Demo completed!"
echo "For Jupyter notebook demo, use: demo/SGDET_on_custom_images.ipynb"