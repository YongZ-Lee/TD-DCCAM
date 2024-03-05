# TD with DCCAM

## Get Started

1. install required packages: including: [Pytorch](https://pytorch.org/) version 1.9.0, [torchvision](https://pytorch.org/vision/stable/index.html) version 0.10.0 and [Timm](https://github.com/rwightman/pytorch-image-models) version 0.5.4

   ```python
   pip install -r requirements.txt
   ```

   For mixed-precision training, please install [apex](https://github.com/NVIDIA/apex)

   For object detection, please additionally install detectron2 library and shapely. Refer to the [Detectron2's INSTALL.md](https://github.com/facebookresearch/detectron2/blob/main/INSTALL.md).

2. Prepare dataset available [ICDAR2019-cTDaR](https://github.com/cndplab-founder/ICDAR2019_cTDaR) 

   Then run `python convert_to_coco_format.py --root_dir=PATH-to-ICDARrepo --target_dir=PATH-toICDAR`. Now the path to processed data is `PATH-to-ICDAR`.

   According to the subset you want to evaluate/fine-tune, a soft link should be created:`ln -s PATH-to-ICDAR/trackA_modern data` or `ln -s PATH-to-ICDAR/at_trackA_archival data`

3. Download pre-trained weights. [here](https://github.com/microsoft/unilm/tree/master/dit)

## Train

Training in ICDAR 2019 cTDaR modern subset:

```python
python train_net.py --config-file ./icdar19_configs/cascade/cascade_dit_base.yaml --num-gpus 1 MODEL.WEIGHTS ./dit-base-224-p16-500k-62d53a.pth OUTPUT_DIR ./icdar19_modern_DCCAM
```

[Detectron2's document](https://detectron2.readthedocs.io/en/latest/tutorials/getting_started.html) may help you for more details.

## Inference

One can run inference using the `inference.py` script.

```python
python inference.py --image_path ./test.jpg --output_file_name ./test-1.jpg --config ./icdar19_configs/cascade/cascade_dit_base.yaml --opts MODEL.WEIGHTS ./icdar19_modern_DCCAM/model_final.pth 
```

## Evaluation

Evaluate model on on ICDAR 2019 cTDaR modern subset: 

```python
python train_net.py --config-file ./icdar19_configs/cascade/cascade_dit_base.yaml --eval-only --num-gpus 1 MODEL.WEIGHTS ./icdar19_modern_DCCAM/model_final.pth OUTPUT_DIR ./eval/icdar19_modern_DCCAM
```

## Acknowledgment

Thanks to [DIT](https://github.com/microsoft/unilm/tree/master/dit) for providing the pre-trained weight framework.

Thanks to [Detectron2](https://github.com/facebookresearch/detectron2) for Cascade Mask R-CNN implementation.

Thanks to [YOLOv8](https://github.com/ultralytics/ultralytics) for providing the open-source framework.
