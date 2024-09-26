import torch
from torch import nn

# 定义一个简单的 Flash Attention 类（此处为示例，实际的 Flash Attention 实现会更复杂）
class FlashAttention(nn.Module):
    def __init__(self, dropout_prob=0.3):  # 设置默认的 Dropout 概率为 0.1
        super(FlashAttention, self).__init__()
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(self, input):
        # 假设这里的 input 是 Flash Attention 的输入
        # 进行 Flash Attention 的计算...

        # 在计算后应用 Dropout
        output = self.dropout(input)
        return output
    
# 创建 FlashAttention 模型实例
model = FlashAttention()

# 模拟输入数据
input_data = torch.randn(1, 2, 10)  # 假设 batch_size=32，序列长度为 10，特征维度为 64
print("input_data:", input_data)

# 前向传播
output = model(input_data)
print("output_data:", output)



# import torch
# from torch import nn

# def dropout_layer(X, dropout):
#     """dropout_layer 函数，该函数以dropout的概率丢弃张量输⼊X中的元素，如上所述重新缩放剩余部分：将剩余部分除以1.0-dropout。"""
#     assert 0 <= dropout <= 1
#     # 在本情况中，所有元素都被丢弃
#     if dropout == 1:
#         return torch.zeros_like(X)

#     # 在本情况中，所有元素都被保留
#     if dropout == 0:
#         return X

#     mask = (torch.randn(X.shape) > dropout).float()

#     return mask * X / (1.0 - dropout)

# X= torch.arange(16, dtype = torch.float32).reshape((2, 8))
# print("test:0", X)
# print(dropout_layer(X, 0.0))
# print("test:0.3", X)
# print(dropout_layer(X, 0.3))
# print("test:0.5", X)
# print(dropout_layer(X, 0.5))
# print("test:1.0", X)
# print(dropout_layer(X, 1.0))

