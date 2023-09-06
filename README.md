# Description

This repo contains code to convert YOLO detection dataset (xywh-format) to YOLO segmentation dataset. The code works on Linux.

For conversion, the Segment Anything Model from Meta is used. Here original repo https://github.com/facebookresearch/segment-anything

# Requirements

Python 3.8 or later with all pyproject.toml dependencies and poetry installed. To install run:

`$ poetry install`

# Usage

`$ python main.py ./dataset --device 0 --weights sam_vit_h_4b8939.pth --model_type vit_h`

The checkpoint for the model can be obtained from this link https://github.com/facebookresearch/segment-anything#model-checkpoints