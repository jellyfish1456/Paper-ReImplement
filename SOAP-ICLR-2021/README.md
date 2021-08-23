论文来自：Online Adversarial Purification based on Self-supervised Learning，ICLR 2021。

实验部分显示，使用 `LC` 生成自监督扰动和 `min-max` 截断提纯样本取得的结果是最好的，论文中也提供了代码，我这里只是精简一下，复现最好的结果。

个人觉得不好的地方（瑕不掩瑜）：在对抗样本提纯阶段，对不同的对抗样本设计了[不同的提纯方案](https://github.com/Mishne-Lab/SOAP/blob/09535124ef13e3f957d25b3a4e54af7f5f713a73/defenses.py)，但实际情况中，是不可能知道对抗样本来自哪种算法的攻击。