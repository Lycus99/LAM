# LAM

![Figure1](https://github.com/Lycus99/LAM/assets/109274751/9cb3753d-4bf2-480e-a8f0-255a1b3e7e92)

Created by Yuchong Li

This repository contains PyTorch implementation for LAM.

We introduce a model LAM to recognize surgical action triplets in the CholecT50 Dataset. 

Our code is based on [Q2L](https://github.com/SlongLiu/query2labels) and [AIM](https://github.com/taoyang1122/adapt-image-models).

The dataset and evaluation metrics are [here](https://github.com/CAMMA-public/cholect50). 

## Results
# CholecTriplet2021 data split

| Model | mAP_I | mAP_V | mAP_T | mAP_IV | mAP_IT | mAP_IVT |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| LAM_Lite | 0.8479 | 0.5526 | 0.4416 | 0.3782 | 0.3991 | 0.3756 |
| LAM_Base | 0.8651 | 0.5571 | 0.4632 | 0.3979 | 0.4323 | 0.4050 |
| LAM_Large | 0.8673 | 0.5605 | 0.4890 | 0.3907 | 0.4404 | 0.4209 |


# Cross-Val data split

| Model | mAP_I | mAP_V | mAP_T | mAP_IV | mAP_IT | mAP_IVT |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| LAM_Lite | 0.932±0.014 | 0.701±0.018 | 0.472±0.021 | 0.464±0.041 | 0.443±0.008 | 0.369±0.022 |
| LAM_Base | 0.936±0.013 | 0.708±0.012 | 0.502±0.029 | 0.477±0.049 | 0.464±0.005 | 0.392±0.020 |
| LAM_Large | 0.946±0.013 | 0.724±0.025 | 0.515±0.040 | 0.490±0.044 | 0.483±0.008 | 0.406±0.022 |

## Pretrained models

The LAM-Lite model used the ResNet-18 as the backbone pre-trained on ImageNet-1K. The LAM-Mega model used the ViT-B/16 and ViT-L/14 based on [CLIP](https://github.com/openai/CLIP)

## Model weights

Our model weights can be downloaded after the paper is accepted.

## Key Files

LAM-Lite model: [https://github.com/Lycus99/LAM/blob/main/lib/models/time_transformer_aim.py#L162](https://github.com/Lycus99/LAM/blob/main/lib/models/time_transformer_aim.py#L162)

build LAM-Lite model: [https://github.com/Lycus99/LAM/blob/main/lib/models/time_transformer_aim.py#L333](https://github.com/Lycus99/LAM/blob/main/lib/models/time_transformer_aim.py#L333)

LAM-Mega model: [https://github.com/Lycus99/LAM/blob/main/lib/models/time_transformer_aim.py#L285](https://github.com/Lycus99/LAM/blob/main/lib/models/time_transformer_aim.py#L285)

build LAM-Mega model: [https://github.com/Lycus99/LAM/blob/main/lib/models/time_transformer_aim.py#L341](https://github.com/Lycus99/LAM/blob/main/lib/models/time_transformer_aim.py#L341)

After training the two models separately, we got the output prediction and then averaged them. 
