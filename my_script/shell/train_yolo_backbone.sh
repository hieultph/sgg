#!/bin/bash

# YOLO Backbone Training for SGG
# This script shows how to train a YOLO backbone before SGG training

echo "Training YOLO backbone for SGG..."

# Step 1: Convert SGG dataset to YOLO format
echo "Converting dataset to YOLO format..."
# Use the provided notebook: process_data/convert_to_yolo.ipynb
# Or run a Python script that does the conversion

# Step 2: Train YOLO model using ultralytics
echo "Training YOLO model..."

# Install ultralytics if not already installed
pip install ultralytics==8.3.100

# Create YOLO training script
cat > train_yolo_backbone.py << 'EOF'
from ultralytics import YOLO
import yaml

# Create dataset config
dataset_config = {
    'path': './datasets/yolo_format',
    'train': 'train',
    'val': 'val', 
    'test': 'test',
    'nc': 150,  # Number of classes for VG150
    'names': [] # List of class names
}

# Save dataset config
with open('yolo_dataset.yaml', 'w') as f:
    yaml.dump(dataset_config, f)

# Initialize YOLO model
model = YOLO('yolov8m.pt')  # or yolov9, yolov10, yolov11, yolov12

# Train the model
results = model.train(
    data='yolo_dataset.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    project='./checkpoints',
    name='yolo_backbone_vg150'
)

print("YOLO backbone training completed!")
print(f"Best weights saved at: {results.save_dir}/weights/best.pt")
EOF

# Run YOLO training
python train_yolo_backbone.py

echo "YOLO backbone training completed!"
echo "Update your SGG config file with the path to the trained weights:"
echo "MODEL.PRETRAINED_DETECTOR_CKPT: './checkpoints/yolo_backbone_vg150/weights/best.pt'"