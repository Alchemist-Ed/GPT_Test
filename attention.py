import tiktoken
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F

torch.manual_seed(1337)
B,T,C = 4, 8, 2
x = torch.randn(B,T,C)


## 原始方法计算多维度向量平均值
xbow = torch.zeros(B,T,C)
for b in range(B):
    for t in range(T):
        ## xprev指代在第b个样本/batch中，截止到第t个单词时，它以及它前面所有的单词
        xprev = x[b,:t+1] ## xprev的形状是[t,C], 它包含了在这个batch中，所有截止到x单词为止，之前所有单词的维度
        xbow[b, t] = torch.mean(xprev, 0)


wei = torch.tril(torch.ones(T, T))
### 矩阵并没有除法运算，pytorch中的矩阵除法运算，实际上时逐元素除法，即先将两个矩阵匹配到同样的形状，再对每个位置上的元素，进行除法
wei = wei / wei.sum(1, keepdim=True)
xbow2 = wei @ x
print(wei)