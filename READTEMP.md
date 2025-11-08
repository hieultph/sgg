## Run training script

```
CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --task predcls --save-best --config-file "configs/VG150/react_yolov8m.yaml" MODEL.ROI_RELATION_HEAD.PREDICTOR REACTPredictor SOLVER.IMS_PER_BATCH 4 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_EPOCH 1 OUTPUT_DIR ./checkpoints/react-precls-exmp

cd /workspace/SGG-Benchmark && conda activate sgg_benchmark && CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10025 --nproc_per_node=1 tools/relation_train_net.py --task predcls --save-best --config-file "configs/VG150/react_yolo11n.yaml" MODEL.ROI_RELATION_HEAD.PREDICTOR REACTPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_EPOCH 1 OUTPUT_DIR ./checkpoints/react-precls-exmp

cd /workspace/SGG-Benchmark && conda activate sgg_benchmark && CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --master_port 10025 --nproc_per_node=1 tools/relation_train_net.py --task predcls --save-best --config-file "configs/VG150/e2e_relation_yolov11n.yaml" MODEL.ROI_RELATION_HEAD.PREDICTOR REACTPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_EPOCH 1 MODEL.PRETRAINED_DETECTOR_CKPT ./checkpoints/VG150/Backbones/yolov11n/weights/best.pt OUTPUT_DIR ./checkpoints/react-precls-exmp
```

## Setup environment

```


conda activate sgg_benchmark && pip install -e . --no-build-isolation


conda activate sgg_benchmark && pip install -e . --no-build-isolation --config-settings editable_mode=compat
```

Use this when build fail

```
conda activate sgg_benchmark && python setup.py clean --all
rm -rf build/ sgg_benchmark.egg-info/ sgg_benchmark/_C*.so
```

---

- change data path in sgg_benchmark/config/paths_catalog.py

========
Not changes

- sgg_benchmark/layers/**init**.py
- sgg_benchmark\layers\dcn\deform_conv_func.py
-
