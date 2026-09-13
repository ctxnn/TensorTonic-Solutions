import torch
import math 

import torch
import math

def attention_scores(q, k, num_heads):
    B, S_q, D = q.shape
    S_k = k.shape[1]

    d_head = D // num_heads

    # split D into heads
    q = q.reshape(B, S_q, num_heads, d_head)
    k = k.reshape(B, S_k, num_heads, d_head)

    # move head dimension before sequence
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    # q: [B, H, S_q, d_head]
    # k: [B, H, S_k, d_head]

    A = q @ k.transpose(-2, -1)

    # A: [B, H, S_q, S_k]

    A = A / math.sqrt(d_head)

    return A
