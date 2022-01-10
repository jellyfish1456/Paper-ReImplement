# Introduction :fire:

论文来自：Adversarial Purification with Score-based Generative Models, ICML, 2021

为什么要复现呢？因为论文的代码只给了测试代码，没有训练代码，且，没有给出模型，所以复现了。

- [模型参考](https://github.com/ermongroup/ncsn/blob/master/models/scorenet.py)，非标准 denoise autoencoder
- [损失参考](https://github.com/ermongroup/ncsn/blob/7f27f4a16471d20a0af3be8b8b4c2ec57c8a0bc1/losses/dsm.py#L18-L26)，叠加噪音时需要注意

注意：在 `eval.py` 中，加载的是训练好的目标模型和保存好的对抗样本，我这里就不提供了，用户应适配自己的模型和数据。