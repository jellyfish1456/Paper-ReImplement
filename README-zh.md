[English](https://github.com/muyuuuu/NRP-pytorch-reimplement/blob/main/README.md) | [简体中文](https://github.com/muyuuuu/NRP-pytorch-reimplement/blob/main/README-zh.md)

# A Self-supervised Approach for Adversarial Robustness

我用 pytorch 复现了这篇[论文](https://ieeexplore.ieee.org/document/9156294)，在[官方仓库](https://github.com/Muzammal-Naseer/NRP)中，只给了训练好的 purifier 模型参数，没有给出完整的训练流程，所以只能复现了。

# 使用

直接这样就行（对应自己的路径）：
```py
python train_purifier.py
```

- `train_purifier.py`，控制训练流程的主文件，使用 `gen_loss` 损失优化 purifier 网络, 使用 `dis_loss` 损失优化判别网络
- `AE.py`，purifier 网络
- `critic.py`，判别网络
- `feature.py`, 特征提取器
- `hybrid.py`，purifier 网络的损失和判别网络的损失
- `ssp.py`，通过最大化特征提取器提取到的特征差异，生成对抗样本

在 `eval` 文件夹下, 加载其它的预训练好的分类模型，验证 purifier 网络是否有效。使用方法如下：

```py
python demo.py
```

# 结果

在训练了 50 个 epoch 后, 获得了 `purifier.pkl`. 然后加载这个模型，在对抗样本进入网络之前使用这个模型对样本进行去噪。注意：所有用于检测 purifier 结果的预训练分类模型都可以从[这里](https://github.com/laisimiao/classification-cifar10-pytorch)下载。

使用了三种攻击算法：FGSM, CW, BIM, 对于每种算法，使用三种扰动阈值来生成对抗样本：ε = 0.015/0.03/0.06.

cifar 10 数据集：

| 模型     | 干净样本的准确率 | FGSM           | CW             | BIM            |
| -------- | ---------------- | -------------- | -------------- | -------------- |
| VGG16    | 79.4             | 76.7/74.9/72.2 | 78.5/78.5/77.5 | 77.4/74.8/71.4 |
| ResNet18 | 78.7             | 74.8/72.5/68.1 | 77.1/76.7/75.8 | 75.6/72.5/68.3 |

cifar 100 dataset 数据集：

| model    | 干净样本的准确率 | FGSM           | CW             | BIM            |
| -------- | ---------------- | -------------- | -------------- | -------------- |
| VGG16    | 65.2             | 65.4/62.1/61.2 | 68.1/68.7/66.3 | 69.3/67.7/59.3 |
| ResNet18 | 65.4             | 69.2/68.5/68.3 | 71.8/70.9/70.3 | 72.1/70.7/66.4 |

ImageNet 数据集：

| model       | 干净样本的准确率 | FGSM | CW   | BIM  |
| ----------- | ---------------- | ---- | ---- | ---- |
| ResNet50    | 66.7             | 62.1 | 66.4 | 59.3 |
| InceptionV3 | 63.8             | 61.0 | 68.4 | 59.0 |