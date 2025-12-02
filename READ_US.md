## Run this to set the python path (lost sometime)
export PYTHONPATH="/mnt/h/gdrive/Takeout/Drive/School/4 Fourth year/BCTN/code/sgg-feedback/sgg:$PYTHONPATH"

## Run webcame demo
### Share screen/webcame on window first
```

```
python demo/webcam_demo.py --config checkpoints/react_PSG/config.yml --weights checkpoints/react_PSG/best_model_epoch_11.pth --dcs 42 --save_path ./output.avi




gst-launch-1.0 mfvideosrc ! videoconvert ! x264enc tune=zerolatency ! rtph264pay config-interval=1 pt=96 ! tcpserversink host=127.0.0.1 port=8554

python convert_psg_to_coco.py --input ../../datasets/psg/psg/psg_train_val.json --output psg_train_val_coco_editable.json

python main_coco.py --output name_of_output_file.json

python ./my_script/data_visualize_web/visualize_dataset_webapp.py

./tools/SGG-Annotate/detect_custom_images.sh

python main_psg.py --psg "../../tools/SGG-Annotate/my_custom_images/custom_psg_detections.json" --images "../../tools/SGG-Annotate/my_custom_images/images"

python main_psg.py --psg "../../datasets/psg/psg/psg_train_val.json" --images "../../datasets/psg/coco/coco/"

tools/SGG-Annotate/my_custom_images/custom_psg_detections.json