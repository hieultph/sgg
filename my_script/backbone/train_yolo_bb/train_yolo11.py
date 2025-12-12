#!/usr/bin/env python3
"""
YOLO11 Training Script - Optimized for A5000 GPU
Dataset: 133 classes detection
Author: Auto-generated
"""

import os
import yaml
import torch
from ultralytics import YOLO
from datetime import datetime
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ============================================
# CONFIGURATION - Optimized for A5000 24GB
# ============================================

# Training Configuration
CONFIG = {
    # ============================================================
    # Model settings
    # ============================================================
    'model': '/workspace/yolo11m.pt',  # dùng absolute path luôn an toàn
    
    # ============================================================
    # Data settings
    # ============================================================
    'data': '/workspace/datasets/YOLO_anno/data.yaml',
    
    # ============================================================
    # Training hyperparameters — SGD-optimized
    # ============================================================
    'epochs': 300,         # Không để quá thấp như paper (20) → YOLO bị underfit
    'batch': 12,           # A5000 24GB xử lý tốt
    'imgsz': 640,
    'patience': 5,         # early stopping với grace period 5 (rất hợp lý)
    
    # ============================================================
    # SGD Optimizer (paper-style)
    # ============================================================
    'optimizer': 'SGD',
    'lr0': 0.01,           # learning rate gốc theo paper
    'lrf': 0.01,           # final LR = 0.01 * 0.01 = 1e-4 (chuẩn cosine decay)
    'momentum': 0.9,       # momentum SGD theo đúng paper
    'weight_decay': 0.0005,
    
    # Warmup — cực kỳ quan trọng cho SGD (tránh gradient explosion)
    'warmup_epochs': 3.0,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,

    # Do SGD thường khó hội tụ → dùng Cosine LR decay
    'cos_lr': True,
    
    # ============================================================
    # Augmentation settings — optimized
    # ============================================================
    # 'hsv_h': 0.015,
    # 'hsv_s': 0.6,
    # 'hsv_v': 0.4,
    # 'degrees': 0.0,
    # 'translate': 0.08,
    # 'scale': 0.5,
    # 'shear': 0.0,
    # 'perspective': 0.0,
    # 'flipud': 0.0,
    # 'fliplr': 0.5,

    # # Mosaic & mixup
    # 'mosaic': 0.7,         # giảm để tăng stability khi dùng SGD
    # 'close_mosaic': 10,
    # 'mixup': 0.0,
    # 'copy_paste': 0.0,
    
    # ============================================================
    # System settings
    # ============================================================
    'workers': 8,
    'device': 0,
    
    # FIX lỗi treo do cache RAM
    'cache': 'disk',

    'amp': True,           # mixed precision giúp tăng tốc
    
    # ============================================================
    # Validation / Logging
    # ============================================================
    'val': True,
    'save_period': 10,
    'plots': True,
    'verbose': True,
}

# Project directories
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
PROJECT_NAME = 'yolo11_133classes'
RUN_NAME = f'train_{TIMESTAMP}'
SAVE_DIR = Path(f'/workspace/runs/{PROJECT_NAME}/{RUN_NAME}')
METRICS_DIR = SAVE_DIR / 'metrics'

# Create directories
METRICS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================
# METRICS LOGGING CLASS
# ============================================

class MetricsLogger:
    """Logger to save all training metrics for research purposes"""
    
    def __init__(self, save_dir):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.epoch_metrics = []
        self.train_history = {
            'epoch': [],
            'train/box_loss': [],
            'train/cls_loss': [],
            'train/dfl_loss': [],
            'metrics/precision(B)': [],
            'metrics/recall(B)': [],
            'metrics/mAP50(B)': [],
            'metrics/mAP50-95(B)': [],
            'val/box_loss': [],
            'val/cls_loss': [],
            'val/dfl_loss': [],
            'lr/pg0': [],
            'lr/pg1': [],
            'lr/pg2': [],
        }
    
    def log_epoch(self, epoch, metrics_dict):
        """Log metrics for each epoch"""
        metrics_dict['epoch'] = epoch
        metrics_dict['timestamp'] = datetime.now().isoformat()
        self.epoch_metrics.append(metrics_dict)
        
        # Update history
        for key in self.train_history.keys():
            if key in metrics_dict:
                self.train_history[key].append(metrics_dict[key])
            elif key == 'epoch':
                self.train_history[key].append(epoch)
    
    def save_metrics(self):
        """Save all metrics to files"""
        # Save epoch-by-epoch metrics as JSON
        json_path = self.save_dir / 'training_metrics.json'
        with open(json_path, 'w') as f:
            json.dump(self.epoch_metrics, f, indent=2)
        
        # Save as CSV for easy analysis
        df = pd.DataFrame(self.epoch_metrics)
        csv_path = self.save_dir / 'training_metrics.csv'
        df.to_csv(csv_path, index=False)
        
        # Save training configuration
        config_path = self.save_dir / 'training_config.json'
        with open(config_path, 'w') as f:
            json.dump(CONFIG, f, indent=2)
        
        print(f"\n✓ Metrics saved to:")
        print(f"  - JSON: {json_path}")
        print(f"  - CSV: {csv_path}")
        print(f"  - Config: {config_path}")
    
    def plot_training_curves(self):
        """Generate training curve plots"""
        df = pd.DataFrame(self.train_history)
        
        if len(df) == 0:
            return
        
        # Set style
        sns.set_style("whitegrid")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('YOLO11 Training Metrics', fontsize=16, fontweight='bold')
        
        # Plot 1: Loss curves
        ax = axes[0, 0]
        if 'train/box_loss' in df.columns:
            ax.plot(df['epoch'], df['train/box_loss'], label='Train Box Loss', linewidth=2)
        if 'val/box_loss' in df.columns:
            ax.plot(df['epoch'], df['val/box_loss'], label='Val Box Loss', linewidth=2)
        if 'train/cls_loss' in df.columns:
            ax.plot(df['epoch'], df['train/cls_loss'], label='Train Cls Loss', linewidth=2)
        if 'val/cls_loss' in df.columns:
            ax.plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training & Validation Losses')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: mAP curves
        ax = axes[0, 1]
        if 'metrics/mAP50(B)' in df.columns:
            ax.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', linewidth=2)
        if 'metrics/mAP50-95(B)' in df.columns:
            ax.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP@0.5:0.95', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('mAP')
        ax.set_title('Mean Average Precision')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Precision & Recall
        ax = axes[1, 0]
        if 'metrics/precision(B)' in df.columns:
            ax.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', linewidth=2)
        if 'metrics/recall(B)' in df.columns:
            ax.plot(df['epoch'], df['metrics/recall(B)'], label='Recall', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Precision & Recall')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Learning Rate
        ax = axes[1, 1]
        if 'lr/pg0' in df.columns:
            ax.plot(df['epoch'], df['lr/pg0'], label='LR pg0', linewidth=2)
        if 'lr/pg1' in df.columns:
            ax.plot(df['epoch'], df['lr/pg1'], label='LR pg1', linewidth=2)
        if 'lr/pg2' in df.columns:
            ax.plot(df['epoch'], df['lr/pg2'], label='LR pg2', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.save_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  - Plot: {plot_path}")


# ============================================
# MAIN TRAINING FUNCTION
# ============================================

def train_yolo11():
    """Main training function"""
    
    print("=" * 60)
    print("YOLO11 Training - Optimized for A5000 GPU")
    print("=" * 60)
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"\n✓ GPU Detected: {gpu_name}")
        print(f"✓ GPU Memory: {gpu_memory:.1f} GB")
    else:
        print("\n⚠ WARNING: No GPU detected! Training will be slow.")
    
    # Print configuration
    print(f"\n📋 Training Configuration:")
    print(f"  Model: {CONFIG['model']}")
    print(f"  Epochs: {CONFIG['epochs']}")
    print(f"  Batch Size: {CONFIG['batch']}")
    print(f"  Image Size: {CONFIG['imgsz']}")
    print(f"  Optimizer: {CONFIG['optimizer']}")
    print(f"  Initial LR: {CONFIG['lr0']}")
    print(f"  Workers: {CONFIG['workers']}")
    print(f"  AMP: {CONFIG['amp']}")
    print(f"  Cache: {CONFIG['cache']}")
    
    # Load data config
    with open(CONFIG['data'], 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"\n📊 Dataset Information:")
    print(f"  Classes: {data_config['nc']}")
    print(f"  Train: {data_config['train']}")
    print(f"  Val: {data_config['val']}")
    print(f"  Test: {data_config['test']}")
    
    print(f"\n💾 Results will be saved to: {SAVE_DIR}")
    
    # Initialize model
    print(f"\n🔄 Loading YOLO11 model: {CONFIG['model']}")
    model = YOLO(CONFIG['model'])
    
    # Initialize metrics logger
    logger = MetricsLogger(METRICS_DIR)
    
    # Start training
    print(f"\n🚀 Starting training...\n")
    
    try:
        results = model.train(
            data=CONFIG['data'],
            epochs=CONFIG['epochs'],
            batch=CONFIG['batch'],
            imgsz=CONFIG['imgsz'],
            patience=CONFIG['patience'],
            
            # Optimizer
            optimizer=CONFIG['optimizer'],
            lr0=CONFIG['lr0'],
            lrf=CONFIG['lrf'],
            momentum=CONFIG['momentum'],
            weight_decay=CONFIG['weight_decay'],
            warmup_epochs=CONFIG['warmup_epochs'],
            warmup_momentum=CONFIG['warmup_momentum'],
            warmup_bias_lr=CONFIG['warmup_bias_lr'],
            
            # Augmentation
            # hsv_h=CONFIG['hsv_h'],
            # hsv_s=CONFIG['hsv_s'],
            # hsv_v=CONFIG['hsv_v'],
            # degrees=CONFIG['degrees'],
            # translate=CONFIG['translate'],
            # scale=CONFIG['scale'],
            # shear=CONFIG['shear'],
            # perspective=CONFIG['perspective'],
            # flipud=CONFIG['flipud'],
            # fliplr=CONFIG['fliplr'],
            # mosaic=CONFIG['mosaic'],
            # mixup=CONFIG['mixup'],
            # copy_paste=CONFIG['copy_paste'],
            # close_mosaic=CONFIG['close_mosaic'],
            
            # System
            workers=CONFIG['workers'],
            device=CONFIG['device'],
            cache=CONFIG['cache'],
            amp=CONFIG['amp'],
            
            # Validation
            val=CONFIG['val'],
            save_period=CONFIG['save_period'],
            plots=CONFIG['plots'],
            verbose=CONFIG['verbose'],
            
            # Project settings
            project=f'/workspace/runs/{PROJECT_NAME}',
            name=RUN_NAME,
            exist_ok=True,
        )
        
        print("\n" + "=" * 60)
        print("✓ Training completed successfully!")
        print("=" * 60)
        
        # Save final results
        print(f"\n📊 Final Results:")
        if hasattr(results, 'results_dict'):
            for key, value in results.results_dict.items():
                print(f"  {key}: {value}")
        
        # Read and save metrics from results.csv
        results_csv = Path(model.trainer.save_dir) / 'results.csv'
        if results_csv.exists():
            df = pd.read_csv(results_csv)
            df = df.rename(columns=lambda x: x.strip())  # Remove whitespace
            
            # Log all epochs
            for idx, row in df.iterrows():
                metrics_dict = row.to_dict()
                logger.log_epoch(idx + 1, metrics_dict)
            
            # Save metrics
            logger.save_metrics()
            logger.plot_training_curves()
        
        # Validate on test set
        print(f"\n🔍 Running validation on test set...")
        test_results = model.val(data=CONFIG['data'], split='test')
        
        # Save test results
        test_metrics = {
            'test/precision': float(test_results.results_dict.get('metrics/precision(B)', 0)),
            'test/recall': float(test_results.results_dict.get('metrics/recall(B)', 0)),
            'test/mAP50': float(test_results.results_dict.get('metrics/mAP50(B)', 0)),
            'test/mAP50-95': float(test_results.results_dict.get('metrics/mAP50-95(B)', 0)),
        }
        
        test_results_path = METRICS_DIR / 'test_results.json'
        with open(test_results_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)
        
        print(f"\n📈 Test Set Results:")
        for key, value in test_metrics.items():
            print(f"  {key}: {value:.4f}")
        print(f"\n✓ Test results saved to: {test_results_path}")
        
        # Save best model path
        best_model = Path(model.trainer.save_dir) / 'weights' / 'best.pt'
        print(f"\n🏆 Best model saved at: {best_model}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        raise


# ============================================
# RESUME TRAINING FUNCTION
# ============================================

def resume_training(checkpoint_path):
    """Resume training from checkpoint"""
    print(f"🔄 Resuming training from: {checkpoint_path}")
    
    model = YOLO(checkpoint_path)
    results = model.train(resume=True)
    
    return results


# ============================================
# MAIN ENTRY POINT
# ============================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLO11 Training Script')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training')
    
    args = parser.parse_args()
    
    if args.resume:
        resume_training(args.resume)
    else:
        train_yolo11()
