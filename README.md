# LAM

![Figure1](https://github.com/Lycus99/LAM/assets/109274751/9cb3753d-4bf2-480e-a8f0-255a1b3e7e92)

Created by Yuchong Li

This repository contains PyTorch implementation for LAM.

We introduce a model LAM to recognize surgical action triplets in the CholecT50 Dataset. 

Our code is based on [Q2L](https://github.com/SlongLiu/query2labels) and [AIM](https://github.com/taoyang1122/adapt-image-models).

The dataset and evaluation metrics are [here](https://github.com/CAMMA-public/cholect50). 

## Pretrained models

The LAM-Lite model used the ResNet-18 as the backbone pre-trained on ImageNet-1K. The LAM-Mega model used the ViT-B/16 and ViT-L/14 based on [CLIP](https://github.com/openai/CLIP)

## Model weights

Our model weights can be downloaded after the paper is accepted.

## Key Files

LAM-Lite model: [https://github.com/Lycus99/LAM/blob/main/lib/models/time_transformer_aim.py#L162](https://github.com/Lycus99/LAM/blob/main/lib/models/time_transformer_aim.py#L162)
