[English]() | [简体中文]()

# A Self-supervised Approach for Adversarial Robustness

I reimplement this [paper](https://ieeexplore.ieee.org/document/9156294) with pytorch. In offical [repository](https://github.com/Muzammal-Naseer/NRP) only the parameters of the pretrained model are provided, but the complete training process is not provided.

# How to use

```py
python train_purifier.py
```

- `train_purifier.py`，The master file that controls the training process, use `gen_loss` to optimize purifier network, use `dis_loss` to optimize critic network
- `AE.py`，the purifier network
- `critic.py`，the critic network
- `feature.py`, the backbone
- `hybrid.py`，the loss of purifier network and critic network
- `ssp.py`，generate adversarial samples by maximizing the perceptual feature difference which is computed by backbone.

In `eval` folder, I load another pretrained model to verify whether purifier network is valid.

```py
python demo.py
```

# Results

After 50 epoch, I get `purifier.pkl`. I load it to remove perturbation before the adversarial samples are fed into the network. Notice, all pretrained model you can download from [here](https://github.com/laisimiao/classification-cifar10-pytorch).

For each attack algorithm(FGSM, CW, BIM), I use three different perturbation thresholds: ε = 0.015/0.03/0.06.

In cifar 10 dataset, here is result:

| model    | clean samples | FGSM           | CW             | BIM            |
| -------- | ------------- | -------------- | -------------- | -------------- |
| VGG16    | 79.4          | 76.7/74.9/72.2 | 78.5/78.5/77.5 | 77.4/74.8/71.4 |
| ResNet18 | 78.7          | 74.8/72.5/68.1 | 77.1/76.7/75.8 | 75.6/72.5/68.3 |

In cifar 100 dataset, here is result:

| model    | clean samples | FGSM           | CW             | BIM            |
| -------- | ------------- | -------------- | -------------- | -------------- |
| VGG16    | 65.2          | 65.4/62.1/61.2 | 68.1/68.7/66.3 | 69.3/67.7/59.3 |
| ResNet18 | 65.4          | 69.2/68.5/68.3 | 71.8/70.9/70.3 | 72.1/70.7/66.4 |

In ImageNet dataset, here is result (ε=0.06):

| model       | clean samples | FGSM | CW   | BIM  |
| ----------- | ------------- | ---- | ---- | ---- |
| ResNet50    | 66.7          | 62.1 | 66.4 | 59.3 |
| InceptionV3 | 63.8          | 61.0 | 68.4 | 59.0 |