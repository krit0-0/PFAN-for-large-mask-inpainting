from torch import nn
import torch
import math
from torch.nn import functional as F
import numpy as np

class TransformerEncoder(nn.Module):
    def __init__(self, patchsizes, num_hidden=256, dis=None):
        super().__init__()
        self.attn = MultiPatchMultiAttention(patchsizes, num_hidden, dis)
        self.feed_forward = FeedForward(num_hidden)

    def forward(self, x, mask=None):
        x = self.attn(x, x, x, mask)
        x = self.feed_forward(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, patchsizes, num_hidden=256, dis=None):
        super().__init__()
        self.cross_attn = MultiPatchMultiAttention(patchsizes, num_hidden, dis=dis)
        self.feed_forward = FeedForward(num_hidden)

    def forward(self, query, key, value):
        x = self.cross_attn(query, key, value)
        x = self.feed_forward(x)
        return x


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    q, k, v  = B * N (h*w) * C
    """

    def forward(self, query, key, value, mask=None, dis=None):
        scores = torch.matmul(query, key.transpose(-2, -1))
        #matmul:矩阵乘法函数，用于计算两个张量的矩阵乘积  transpose:返回矩阵 key 的转置
        if dis is not None:

            scores = scores + dis

        scores = scores / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
            #scores = scores * mask
        p_attn = torch.nn.functional.softmax(scores, dim=-1)
        p_val = torch.matmul(p_attn, value)
        return p_val, p_attn


class MultiAttn(nn.Module):
    """
    Attention Network
    """

    def __init__(self, head=4):
        """
        :param num_hidden: dimension of hidden
        :param h: num of heads
        key, value, query B*C*H*W
        """
        super().__init__()
        self.h = head

        self.attn = Attention()

    def forward(self, query, key, value, mask=None, dis=None):

        B,N,C = key.size()
        num_hidden_per_attn = C // self.h
        k = key.view(B, N, self.h, num_hidden_per_attn).contiguous()
        v = value.view(B, N, self.h, num_hidden_per_attn).contiguous()
        q = query.view(B, N, self.h, num_hidden_per_attn).contiguous()

        k = k.permute(2,0,1,3).contiguous() # view(-1, N, num_hidden_per_attn)
        v = v.permute(2,0,1,3).contiguous()
        q = q.permute(2,0,1,3).contiguous()

        if mask is not None:
            mask = mask.unsqueeze(0)
            out, attn = self.attn(q, k, v, mask, dis)
        else:
            out, attn = self.attn(q, k, v, dis=dis)
        out = out.view(self.h, B, N, num_hidden_per_attn)
        out = out.permute(1, 2, 0, 3).contiguous().view(B, N, C).contiguous()
        return out, attn


class FeedForward(nn.Module):
    def __init__(self, num_hidden):
        super(FeedForward, self).__init__()
        # We set d_ff as a default to 2048
        self.conv = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=2, dilation=2),
            nn.InstanceNorm2d(num_hidden, track_running_stats=False),    #对每个样本的每个通道进行归一化
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1),
            nn.InstanceNorm2d(num_hidden, track_running_stats=False),
            nn.LeakyReLU(0.2, inplace=True))

    def forward(self, x):
        x = x + self.conv(x)
        return x


class MultiPatchMultiAttention(nn.Module):
    def __init__(self, patchsize, num_hidden, dis):
        super().__init__()
        self.ps = patchsize
        num_head = patchsize * 4
        self.query_embedding = nn.Conv2d(
            num_hidden, num_hidden, kernel_size=1, padding=0)
        self.value_embedding = nn.Conv2d(
            num_hidden, num_hidden, kernel_size=1, padding=0)
        self.key_embedding = nn.Conv2d(
            num_hidden, num_hidden, kernel_size=1, padding=0)

        self.output_linear = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True))


        self.attention = MultiAttn(head=num_head)
        length = 64 // patchsize
        dis = dis.view(length * length, length * length).float()
        self.register_buffer('lap', dis)
        a = nn.Parameter(torch.ones(1))
        setattr(self, 'lap_a', a)

    def forward(self, query, key, value, mask=None):
        residual = query
        B, C, H, W = query.size()
        q = self.query_embedding(query)
        k = self.key_embedding(key)
        v = self.value_embedding(value)

        num_w = W // self.ps
        num_h = H // self.ps
        # 1) embedding and reshape

        #.contiguous():返回一个连续的内存块版本的张量。
        q = q.view(B, C, num_h, self.ps, num_w, self.ps).contiguous()    # B * C* h/s * s * w/s * s
        k = k.view(B, C, num_h, self.ps, num_w, self.ps).contiguous()
        v = v.view(B, C, num_h, self.ps, num_w, self.ps).contiguous()
        # B * (h/s * w/s) * (C * s * s)

        #positional_embedding
        q += nn.Parameter(torch.randn_like(q))
        k += nn.Parameter(torch.randn_like(k))
        v += nn.Parameter(torch.randn_like(v))

        q = q.permute(0, 2, 4, 1, 3, 5).contiguous().view(B,  num_h*num_w, C * self.ps * self.ps)
        k = k.permute(0, 2, 4, 1, 3, 5).contiguous().view(B,  num_h*num_w, C * self.ps * self.ps)
        v = v.permute(0, 2, 4, 1, 3, 5).contiguous().view(B,  num_h*num_w, C * self.ps * self.ps)
        dis = F.softplus(self.lap_a) * self.lap
        if mask is not None:
            m = mask.view(B, 1, num_h, self.ps, num_w, self.ps).contiguous()
            m = m.permute(0, 2, 4, 1, 3, 5).contiguous().view(B, num_h * num_w, self.ps * self.ps).contiguous()
            m = (m.mean(-1) < 0.5).unsqueeze(1).repeat(1, num_w * num_h, 1)
            result, _ = self.attention(q, k, v, m, dis)

        else:
            result, _ = self.attention(q, k, v, dis=dis)
        # 3) "Concat" using a view and apply a final linear.
        result = result.view(B, num_h, num_w, C, self.ps,  self.ps).contiguous()
        result = result.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, C, H, W).contiguous()
        output = self.output_linear(result)
        output = output + residual
        return output
