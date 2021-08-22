# 一些顶会论文的复现

1. 在自己写论文的时候，难免和别人的算法进行对比，有的时候模型不一样、数据不一样无法直接对比，必须要复现算法。
2. 但是有的论文不仅不提供源程序，且不给出模型结构，要么实验部分什么都不说，要么在实验部分有很多玄学的参数设置与诡异的说明。
3. 甚至还有人只给出预测程序，而不提供训练模型的程序，或者不提供一些中间数据，更有甚者文中算法描述与代码并不对应，结果存疑，懂得都懂。
4. 大多时候我也只能尽力理解论文的意思，然后写出代码。
5. 另外，有些地方我会加载本地的模型和数据，路径肯定对不上，且，我也不想传一堆数据到 github 上，自行替换即可。本仓库的重点是写出论文的算法。

如果无特殊声明，复现的都是「不提供源码、或不提供实验设置、或论文算法与提供代码不符」的论文。

## 对抗样本去噪相关

- [A Self-supervised Approach for Adversarial Robustness](https://arxiv.org/abs/2006.04924)，CVPR 2020，对应 NRP-CVPR-2020。
- [Adversarial purification with Score-based generative models](https://arxiv.org/abs/2106.06041)，ICML 2021，对应 ADP-ICML-2021。
- [Online Adversarial Purification based on Self-supervised Learning](https://openreview.net/forum?id=_i3ASPp12WS)，ICLR 2021，这篇论文提供了代码，代码写的挺不错的，且和论文算法一致。我这里只是精简了一下，对应 SOAP-ICLR-2021。