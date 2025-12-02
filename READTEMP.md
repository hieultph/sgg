## Run training script

```
CUDA_VISIBLE_DEVICES=0 python tools/relation_train_net.py --task predcls --save-best --config-file "configs/VG150/react_yolov8m.yaml" MODEL.ROI_RELATION_HEAD.PREDICTOR REACTPredictor SOLVER.IMS_PER_BATCH 4 TEST.IMS_PER_BATCH 1 DTYPE "float16" SOLVER.MAX_EPOCH 1 OUTPUT_DIR ./checkpoints/react-precls-exmp

cd /workspace/SGG-Benchmark && conda activate sgg_benchmark && CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --master_port 10025 --nproc_per_node=1 tools/relation_train_net.py --task predcls --save-best --config-file "configs/VG150/react_yolov8m.yaml" MODEL.ROI_RELATION_HEAD.PREDICTOR REACTPredictor SOLVER.IMS_PER_BATCH 12 TEST.IMS_PER_BATCH 2 DTYPE "float16" SOLVER.MAX_EPOCH 1 OUTPUT_DIR ./checkpoints/react-precls-exmp

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


# MAI
============================

export PYTHONPATH="/mnt/h/gdrive/Takeout/Drive/School/4 Fourth year/BCTN/code/sgg-feedback/sgg:$PYTHONPATH"

python demo/webcam_demo.py --config checkpoints/react_PSG/config.yml --weights checkpoints/react_PSG/best_model_epoch_11.pth --dcs 42 --save_path ./output.avi

gst-launch-1.0 mfvideosrc ! videoconvert ! x264enc tune=zerolatency bitrate=2000 speed-preset=superfast ! rtph264pay ! udpsink host=127.0.0.1 port=8090


gst-launch-1.0 mfvideosrc ! videoconvert ! x264enc tune=zerolatency speed-preset=superfast bitrate=2000 ! rtph264pay config-interval=1 name=pay0 pt=96 ! gdppay ! tcpserversink host=0.0.0.0 port=8554



gst-launch-1.0 mfvideosrc ! videoconvert ! x264enc tune=zerolatency ! rtph264pay config-interval=1 pt=96 ! tcpserversink host=127.0.0.1 port=8554

python convert_psg_to_coco.py --input ../../datasets/psg/psg/psg_train_val.json --output psg_train_val_coco_editable.json

python main_coco.py --output name_of_output_file.json

python visualize_dataset_webapp.py