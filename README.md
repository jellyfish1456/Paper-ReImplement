**大多时候我也只能尽力理解论文的意思，然后写出代码。**

# 一些顶会论文的复现

在自己写论文的时候，难免和别人的算法进行对比，有的时候模型不一样、数据不一样无法直接对比，必须要复现算法。

但是有的论文不仅不提供源程序，且不给出模型结构，甚至实验部分什么都不说，要么在实验部分有很多玄学的参数设置与诡异的说明，或者只给出预测程序，而不提供训练模型的程序，或者不提供一些中间数据。

## 本仓库注意事项

有些地方我会加载本地的模型和数据，路径肯定对不上，且，我也不想上传一堆数据到 github 上，自行替换即可。本仓库的重点是写出论文的算法。

如果无特殊声明，复现的都是「不提供完整源码、或不提供实验设置」的论文。因为有人提供不完整代码，因此**发现论文和官方代码不一致时，以官方代码为主。**

## 对抗样本去噪相关

- CVPR 2019，[ComDefend: An Efficient Image Compression Model to Defend Adversarial Examples
](https://arxiv.org/abs/1811.12673)，对应 2019-CVPR-Comdefend，效果和论文一致。
- CVPR 2020，[A Self-supervised Approach for Adversarial Robustness](https://arxiv.org/abs/2006.04924)，对应 2020-CVPR-NRP，效果和论文一致。
- ICML 2021，[Adversarial purification with Score-based generative models](https://arxiv.org/abs/2106.06041)，对应 2021-ICML-ADP，效果和论文相差较远。自己复现的话需要注意：这个论文模型结构参考了另一篇论文。
- ICLR 2021，[Online Adversarial Purification based on Self-supervised Learning](https://openreview.net/forum?id=_i3ASPp12WS)，这篇论文提供了代码，代码写的挺不错的，且和论文算法一致。我这里只是精简了一下，对应 2021-ICLR-SOAP，效果和论文一致。

## 对抗训练相关

- AAAI 2021，[Understanding Catastrophic Overfitting in Single-step Adversarial Training](https://arxiv.org/abs/2010.01799)，对应 2021-AAAI-CO。这篇论文的作者开发了一个大名鼎鼎的 `torchattacks` 库，论文也用了这个库。但是这个库的更新导致的论文的代码无法跑通，所以复现。我还是很看好这篇论文的，不仅把自己的想法说的很清楚，解决了问题，甚至把前人没解释清楚的东西帮他解释了，不得不说佩服。效果和论文一致。
