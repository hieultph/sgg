## 1. Train the YOLO backbone

### Using `convert_to_yolo.ipynb` to convert file in SGG format (.h5) to YOLO format for object detection training

Link:
https://github.com/Maelic/SGG-Benchmark/blob/main/process_data/convert_to_yolo.ipynb

### Modify `e2e_relation_yolov8m.yaml` point to your YOLO model

Once you have a model, you can modify this config file and change the path **PRETRAINED_DETECTOR_CKPT** to your model weights. Please note that you will also need to change the variable **SIZE** and **OUT_CHANNELS** accordingly if you use another variant of YOLO (nano, small or large for instance).

For training an SGG model with YOLO as a backbone, you need to modify the **META_ARCHITECTURE** variable in the same config file to **GeneralizedYOLO**. You can then follow the standard procedure for PREDCLS, SGCLS or SGDET training below.

## 2. Perform training on Scene Graph Generation

For Predicate Classification (PredCls), we need to set:

```
--task predcls
```

### Predefined Models

For REACT Model:

```
MODEL.ROI_RELATION_HEAD.PREDICTOR REACTPredictor
```

The default settings are under `configs/e2e_relation_X_101_32_8_FPN_1x.yaml` and `sgg_benchmark/config/defaults.py`. The priority is `command > yaml > defaults.py`

### Customize Your Own Model

If you want to customize your own model, you can refer sgg_benchmark/modeling/roi_heads/relation_head/model_XXXXX.py and sgg_benchmark/modeling/roi_heads/relation_head/utils_XXXXX.py. You also need to add the corresponding nn.Module in sgg_benchmark/modeling/roi_heads/relation_head/roi_relation_predictors.py. Sometimes you may also need to change the inputs & outputs of the module through sgg_benchmark/modeling/roi_heads/relation_head/relation_head.py.

### Examples of the Training Command

SOLVER.MAX_EPOCH

--save-best

```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --master_port 10025 --nproc_per_node=2 tools/relation_train_net.py --task predcls --save-best --config-file "configs/e2e_relation_X_101_32_8_FPN_1x.yaml" MODEL.ROI_RELATION_HEAD.PREDICTOR MotifPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_EPOCH 20 MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/pretrained_faster_rcnn/model_final.pth OUTPUT_DIR ./checkpoints/motif-precls-exmp
```

where **MODEL.PRETRAINED_DETECTOR_CKPT** is the pretrained Faster R-CNN model you want to load, **OUTPUT_DIR** is the output directory used to save checkpoints and the log. Since we use the **WarmupReduceLROnPlateau** as the learning scheduler for SGG, **SOLVER.STEPS** is not required anymore.

### Hyperparameters Tuning

```
pip install ray[data,train,tune] optuna tensorboard
```

We provide a training loop for hyperparameters tuning in `hyper_param_tuning`.py. This script uses the RayTune library for efficient hyperparameters search. You can define a **search_space** object with different values related to the optimizer (AdamW and SGD supported for now) or directly customize the model structure with model parameters (for instance Linear layers dimensions or MLP dimensions etc). The **ASHAScheduler** scheduler is used for the early stopping of bad trials. The default value to optimize is the overall loss but this can be customize to specific loss values or standard metrics such as mean_recall

```
CUDA_VISIBLE_DEVICES=0 python tools/hyper_param_tuning.py --save-best --task sgdet --config-file "./configs/IndoorVG/e2e_relation_yolov10.yaml" MODEL.ROI_RELATION_HEAD.PREDICTOR PrototypeEmbeddingNetwork DTYPE "float16" SOLVER.PRE_VAL True GLOVE_DIR /home/maelic/glove OUTPUT_DIR ./checkpoints/IndoorVG4/SGDET/penet-yolov10m SOLVER.IMS_PER_BATCH 8
```

**To watch the results with tensorboardX:**
tensorboard --logdir=./ray_results/train_relation_net_2024-06-23_15-28-01

### Evaluation
