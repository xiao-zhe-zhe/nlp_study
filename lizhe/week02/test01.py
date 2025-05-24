from __future__ import print_function
import torch

# 创建一个没有初始化的矩阵
x = torch.empty(5,3)
print(x)

x = torch.randn(5,3)
print(x)