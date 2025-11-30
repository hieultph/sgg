# Hướng dẫn Train tiếp (Resume Training) từ Checkpoint

## 📋 Tổng quan

Bạn có checkpoint: `checkpoints/react_PSG/best_model_epoch_11.pth`

Để train tiếp từ checkpoint này, codebase hỗ trợ tự động resume training nếu checkpoint được đặt đúng nơi.

## 🔧 Cách 1: Auto Resume (Khuyến nghị)

### Bước 1: Copy checkpoint vào OUTPUT_DIR

```bash
# Xem OUTPUT_DIR trong config
# Mặc định: './output/relation_baseline'

# Tạo thư mục nếu chưa có
mkdir -p output/react_PSG_resume

# Copy checkpoint vào đó với tên chuẩn
cp checkpoints/react_PSG/best_model_epoch_11.pth output/react_PSG_resume/model_final.pth

# HOẶC tạo symlink
ln -s $(pwd)/checkpoints/react_PSG/best_model_epoch_11.pth output/react_PSG_resume/model_final.pth
```

### Bước 2: Chỉnh config

Tạo file config mới `configs/PSG/react_yolov8m_resume.yaml`:

```yaml
_BASE_: "react_yolov8m.yaml"

SOLVER:
  MAX_EPOCH: 30                    # Train thêm đến epoch 30 (hiện tại đã có 11)
  BASE_LR: 0.001                   # Giảm learning rate (1/10 của ban đầu)
  WARMUP_FACTOR: 1.0               # Không cần warmup nữa
  UPDATE_SCHEDULE_DURING_LOAD: True  # Quan trọng! Update scheduler state

OUTPUT_DIR: './output/react_PSG_resume'

# Optional: Nếu muốn fine-tune với learning rate khác nhau
# SOLVER:
#   BASE_LR: 0.001
#   STEPS: (5000, 10000)
```

### Bước 3: Train

```bash
conda activate sgg_benchmark

# Single GPU
CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py \
  --config-file configs/PSG/react_yolov8m_resume.yaml

# Multi GPU (nếu có)
python -m torch.distributed.launch \
  --nproc_per_node=2 \
  tools/relation_train_net.py \
  --config-file configs/PSG/react_yolov8m_resume.yaml
```

### Code logic trong `relation_train_net.py`:

```python
# Dòng 161-163
if checkpointer.has_checkpoint():
    # Tự động load checkpoint nếu tìm thấy trong OUTPUT_DIR
    extra_checkpoint_data = checkpointer.load(
        checkpointer.get_checkpoint_file(), 
        update_schedule=cfg.SOLVER.UPDATE_SCHEDULE_DURING_LOAD
    )
    arguments.update(extra_checkpoint_data)  # Load epoch, iteration, optimizer state
```

## 🎯 Cách 2: Load từ đường dẫn cụ thể

Nếu không muốn copy file, sửa code trong `tools/relation_train_net.py`:

```python
# Thay thế dòng 161-163
checkpoint_path = "checkpoints/react_PSG/best_model_epoch_11.pth"
if os.path.exists(checkpoint_path):
    extra_checkpoint_data = checkpointer.load(
        checkpoint_path, 
        with_optim=True,  # Load optimizer state
        update_schedule=True  # Update scheduler
    )
    arguments.update(extra_checkpoint_data)
    logger.info(f"Resumed from checkpoint: {checkpoint_path}")
    logger.info(f"Starting from epoch {arguments.get('epoch', 0)}")
```

## 📊 Checkpoint Structure

Checkpoint PSG thường chứa:

```python
{
    'model': OrderedDict(...),        # Model weights
    'optimizer': {...},                # Optimizer state (Adam/SGD state)
    'scheduler': {...},                # Learning rate scheduler state
    'epoch': 11,                       # Current epoch
    'iteration': XXXX,                 # Current iteration
    'best_metric': 0.XXX,             # Best validation metric
}
```

## ⚙️ Các tham số quan trọng khi resume

### 1. Learning Rate Strategy

**Option A: Giảm LR (Fine-tuning)**
```yaml
SOLVER:
  BASE_LR: 0.001  # 1/10 của original (0.01)
  WARMUP_FACTOR: 1.0
```

**Option B: Tiếp tục với LR schedule cũ**
```yaml
SOLVER:
  BASE_LR: 0.01
  UPDATE_SCHEDULE_DURING_LOAD: True  # Quan trọng!
```

### 2. Training Epochs

```yaml
SOLVER:
  MAX_EPOCH: 30  # Train thêm 19 epochs nữa (từ 11 → 30)
```

### 3. Validation & Checkpoint

```yaml
SOLVER:
  VAL_PERIOD: 2000       # Validate mỗi 2000 iterations
  CHECKPOINT_PERIOD: 2000  # Save checkpoint mỗi 2000 iterations
```

## 🔍 Kiểm tra checkpoint đang load

Thêm vào `relation_train_net.py` sau dòng load checkpoint:

```python
if checkpointer.has_checkpoint():
    extra_checkpoint_data = checkpointer.load(...)
    arguments.update(extra_checkpoint_data)
    
    # Debug info
    logger.info("="*80)
    logger.info("📦 RESUMING FROM CHECKPOINT")
    logger.info(f"  ✓ Checkpoint: {checkpointer.get_checkpoint_file()}")
    logger.info(f"  ✓ Starting Epoch: {arguments.get('epoch', 0)}")
    logger.info(f"  ✓ Starting Iteration: {arguments.get('iteration', 0)}")
    logger.info(f"  ✓ Best Metric: {arguments.get('best_metric', 0.0):.4f}")
    logger.info("="*80)
```

## 🎓 Tips & Best Practices

### 1. **Fine-tuning vs Continue Training**

**Continue Training** (tiếp tục từ epoch 11):
- Giữ nguyên learning rate schedule
- `UPDATE_SCHEDULE_DURING_LOAD: True`
- Phù hợp nếu training bị gián đoạn

**Fine-tuning** (điều chỉnh model):
- Giảm learning rate (×0.1 hoặc ×0.01)
- `WARMUP_FACTOR: 1.0` (bỏ warmup)
- Phù hợp nếu muốn cải thiện model trên data mới

### 2. **Freeze Backbone (Optional)**

Nếu chỉ muốn train relation head:

```yaml
MODEL:
  BACKBONE:
    FREEZE: True  # Đã có trong config
```

### 3. **Monitoring Training**

Sử dụng Weights & Biases:

```bash
python tools/relation_train_net.py \
  --config-file configs/PSG/react_yolov8m_resume.yaml \
  --use_wandb \
  --project_name "PSG_SGG_Resume"
```

### 4. **Validation trước khi train**

Kiểm tra metric hiện tại:

```bash
python tools/relation_test_net.py \
  --config-file configs/PSG/react_yolov8m_resume.yaml
```

## 🚀 Quick Start Script

```bash
#!/bin/bash
# resume_training.sh

# Tạo output directory
mkdir -p output/react_PSG_resume

# Copy checkpoint
cp checkpoints/react_PSG/best_model_epoch_11.pth \
   output/react_PSG_resume/model_final.pth

# Activate conda
conda activate sgg_benchmark

# Train
CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py \
  --config-file configs/PSG/react_yolov8m_resume.yaml \
  2>&1 | tee logs/resume_training_$(date +%Y%m%d_%H%M%S).log
```

## 📈 Expected Behavior

Khi resume thành công, log sẽ hiển thị:

```
Loading checkpoint from output/react_PSG_resume/model_final.pth
✓ Model weights loaded
✓ Optimizer state loaded
✓ Scheduler state loaded
Starting from epoch: 11
Starting from iteration: XXXXX
Best metric so far: 0.XXXX

Epoch 12/30 | Loss=X.XX | mR@20=0.XXX
...
```

## ⚠️ Troubleshooting

### Lỗi: "Model architecture mismatch"

**Nguyên nhân:** Config khác với lúc train checkpoint

**Giải pháp:** Đảm bảo config giống nhau (số classes, predictor type, etc.)

### Lỗi: "Optimizer state mismatch"

**Nguyên nhân:** Learning rate hoặc optimizer type khác

**Giải pháp:** 
```python
checkpointer.load(checkpoint_path, with_optim=False)  # Không load optimizer
```

### Checkpoint không được load

**Nguyên nhân:** Tên file không đúng

**Giải pháp:** Checkpoint phải tên `model_final.pth` hoặc `model_XXXXXXX.pth`

---

**Tóm lại:** Chỉ cần copy checkpoint vào OUTPUT_DIR với tên `model_final.pth`, code sẽ tự động resume! 🎉
