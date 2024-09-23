import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def alibi_linear_biases(n_heads):
    slopes = torch.pow(torch.tensor(2.0), torch.tensor(-8.0 / n_heads))
    print(f"Slopes shape: {slopes.shape}")
    print(f"Slopes: {slopes}")
    arange_tensor = torch.arange(1, n_heads + 1).view(1, n_heads, 1, 1)
    print(f"Arange tensor shape: {arange_tensor.shape}")
    alibi = slopes * arange_tensor.view(1, n_heads, 1, 1)
    print(f"Alibi shape: {alibi.shape}")
    return alibi

def print_tensor_according_position(tensor, position):
    print(tensor.flatten()[:position])

def flash_attention_with_alibi(query, key, value, alibi):
    batch_size, n_heads, seq_length, head_dim = query.shape
    scale = head_dim ** -0.5
    qk = torch.matmul(query, key.transpose(-2, -1)) * scale

    print(f"qk shape: {qk.shape}")
    # Add ALiBi biases
    print(f"test qk:")
    print_tensor_according_position(qk, 10)
    qk += alibi.expand(batch_size, n_heads, seq_length, seq_length)
    print(f"test after add alibi qk:")
    print_tensor_according_position(qk, 10)
    attn_scores = F.softmax(qk, dim=-1)
    attn_output = torch.matmul(attn_scores, value)
    print(f"attn_output shape: {attn_output.shape}")
    return attn_output

# 假设输入形状为 (batch_size, num_heads, seq_length, head_dim)
batch_size = 2
num_heads = 8
seq_length = 16
head_dim = 64
query = torch.randn(batch_size, num_heads, seq_length, head_dim)
key = torch.randn(batch_size, num_heads, seq_length, head_dim)
value = torch.randn(batch_size, num_heads, seq_length, head_dim)

alibi = alibi_linear_biases(num_heads)
print("alibli:", alibi)
attn_output = flash_attention_with_alibi(query, key, value, alibi)
print(attn_output.shape)
print(attn_output)
