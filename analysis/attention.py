import torch
from torch import einsum, nn


def _scripted_modulation(q, kv_mod, B: int, N: int, num_heads: int, head_dim: int):
    return (
        q.view(B, num_heads, N, 1, head_dim)
        @ (q @ kv_mod.view(B, num_heads, head_dim, int(head_dim * (head_dim + 1)))).view(
            B, num_heads, N, head_dim, head_dim + 1
        )
    ).view(B, num_heads, N, head_dim + 1)


@torch.jit.script
def box_tensor(a, b):
    return (a.unsqueeze(-1) * b.unsqueeze(-2)).view(a.shape[:-1] + [-1])


class LinearTaylorAttention(nn.Module):
    def forward(self, qkv):
        q, k, v = qkv.unbind(0)
        B, H, N, d = q.shape
        # q = torch.nn.functional.normalize(q, dim=-1) * self.temparature
        # k = torch.nn.functional.normalize(k, dim=-1) / self.scale
        v = torch.cat([torch.ones(*v.shape[:-1], 1, device=v.device, dtype=v.dtype), v], dim=-1)
        kv_mod = box_tensor(k, k).transpose(-1, -2) @ v
        y = 0.5 * box_tensor(q, q) @ kv_mod + q @ (k.transpose(-1, -2) @ v) + v.sum(-2).view(B, H, 1, d + 1)

        y_norm, y = y[:, :, :, :1], y[:, :, :, 1:]
        return y / y_norm  # B x H x N x d

    def forward_old(self, QKV):
        Q, K, V = QKV.unbind(0)
        B, H, N, d = Q.shape
        V = torch.cat([torch.ones(*V.shape[:-1], 1, device=V.device, dtype=V.dtype), V], dim=-1)
        kv_mod = einsum("...ni,...nj,...nk->...ijk", K, K, V)  # B, H, d, d, d
        y = (
            0.5 * _scripted_modulation(Q, kv_mod, B, N, H, d)
            + Q @ (K.transpose(-1, -2) @ V)
            + V.sum(-2).view(B, H, 1, d + 1)
        )
        y_norm, y = y[:, :, :, :1], y[:, :, :, 1:]
        return y / y_norm  # B x H x N x d


@torch.jit.script
def taylor_exp(x):
    return x**2 / 2 + x + 1


class SquaredTaylorAttention(nn.Module):
    def forward(self, QKV):
        Q, K, V = QKV.unbind(0)
        attn = Q @ K.transpose(-2, -1)
        attn = 0.5 * attn.square() + attn + 1
        # attn = taylor_exp(attn)
        attn = attn / attn.sum(-1).view(*attn.shape[:-1], 1)
        return attn @ V


class BaselineAttention(nn.Module):
    def forward(self, QKV):
        Q, K, V = QKV.unbind(0)
        attn = Q @ K.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        return attn @ V


class SepLinearTaylorAttention(nn.Module):
    def forward(self, qkv):
        q, k, v = qkv.unbind(0)
        B, H, N, d = q.shape
        # q = torch.nn.functional.normalize(q, dim=-1) * self.temparature
        # k = torch.nn.functional.normalize(k, dim=-1) / self.scale
        # v = torch.cat([torch.ones(*v.shape[:-1], 1, device=v.device, dtype=v.dtype), v], dim=-1)
        kv_mod = box_tensor(k, k).transpose(-1, -2) @ v
        y = 0.5 * box_tensor(q, q) @ kv_mod + q @ (k.transpose(-1, -2) @ v) + v.sum(-2).view(B, H, 1, d)

        k = k.sum(dim=-2)

        y_norm = 0.5 * box_tensor(q, q) @ box_tensor(k, k).unsqueeze(-1) + +q @ k.unsqueeze(-1) + 1

        # y_norm, y = y[:, :, :, :1], y[:, :, :, 1:]
        return y / y_norm.view(B, H, N, 1)  # B x H x N x d
