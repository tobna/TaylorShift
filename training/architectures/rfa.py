# Implementation adapted from information in https://arxiv.org/pdf/2103.02143.pdf
import logging
from functools import partial
from math import sqrt

import torch
from timm.models import register_model
from torch import nn
from timm.models.vision_transformer import Attention, Block
from torch.nn import functional as F

from architectures.transformer import Transformer
from architectures.vit import TimmViT
from resizing_interface import vit_sizes


class RandomMatrixHolder(nn.Module):
    def __init__(
        self, head_dim, num_vectors=64, num_matrices=200, redraw_on_call=False
    ):
        # authors suggest presampling 200 random matrices
        # authors use 64 random vectors per matrix -> Random Feature Map Size
        super().__init__()
        self.head_dim = head_dim
        self.num_vectors = num_vectors
        self.num_matrices = num_matrices
        self.redraw_on_call = redraw_on_call
        if not redraw_on_call:
            self.random_matrices = torch.randn(num_matrices, head_dim, num_vectors)

    def __getitem__(self, item):
        if self.redraw_on_call:
            return self.get_random(1, redraw=True)
        return self.random_matrices[item]

    def get_random(self, n=1, device=None, redraw=None):
        if redraw is None:
            redraw = self.redraw_on_call
        if redraw:
            return torch.randn(n, self.head_dim, self.num_vectors, device=device)
        random_indices = torch.randint(0, self.num_matrices, (n,), device=device)
        self.random_matrices = self.random_matrices.to(device)
        if n == 1:
            return self.random_matrices[random_indices[0]].to(device)
        return self.random_matrices[random_indices].to(device)


class RFAttention(Attention):
    def __init__(self, *args, matrix_holder=None, qk_norm=False, **kwargs):
        super().__init__(*args, qk_norm=False, **kwargs)
        if matrix_holder is None:
            logging.warning(
                "No matrix holder provided. It's better to share random matrices between layers."
            )
            self.matrix_holder = RandomMatrixHolder(self.head_dim)
        assert not qk_norm, "qk_norm is not supported by RFA"
        self.matrix_holder = matrix_holder
        self.sigma = nn.Parameter(torch.ones(self.num_heads, self.head_dim))

    def _rfa_kernel(self, x):
        # x is (B x) H x N x D
        B, H, N, d = x.shape
        # print(self.matrix_holder.random_matrices.device, self.sigma.device, self.matrix_holder.get_random(2).device)
        random_features = self.sigma.unsqueeze(-1).unsqueeze(
            0
        ) * self.matrix_holder.get_random(B * H, device=x.device).view(B, H, d, -1)
        # random_features is H x D x V
        x = x @ random_features
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        return x / sqrt(self.matrix_holder.num_vectors)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # q, k, v are B x H x N x D
        q, k = F.normalize(q, p=2.0, dim=-1), F.normalize(k, p=2.0, dim=-1)

        q = self._rfa_kernel(q)  # q is B x H x N x 2V
        k = self._rfa_kernel(k)  # k is B x H x N x 2V

        kv = k.transpose(-1, -2) @ v  # kv is B x H x 2V x D
        z = k.sum(dim=-2).unsqueeze(-1)  # z is B x H x 2V

        kv = self.attn_drop(kv)

        x = (q @ kv) / (q @ z)  # x is B x H x N x D

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RFABlock(Block):
    def __init__(
        self,
        dim,
        num_heads,
        matrix_holder=None,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        **kwargs,
    ):
        super().__init__(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            **kwargs,
        )
        self.attn = RFAttention(
            dim,
            matrix_holder=matrix_holder,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )


class RFAViT(TimmViT):
    def __init__(
        self,
        *args,
        embed_dim=768,
        num_heads=12,
        num_vectors=64,
        num_matrices=200,
        **kwargs,
    ):
        matrix_holder = RandomMatrixHolder(
            embed_dim // num_heads,
            num_vectors=num_vectors,
            num_matrices=num_matrices,
            redraw_on_call=True,
        )
        super().__init__(
            *args,
            embed_dim=embed_dim,
            num_heads=num_heads,
            block_fn=partial(RFABlock, matrix_holder=matrix_holder),
            **kwargs,
        )


class RFATransformer(Transformer):
    def __init__(
        self,
        *args,
        embed_dim=768,
        num_heads=12,
        num_vectors=64,
        num_matrices=200,
        **kwargs,
    ):
        matrix_holder = RandomMatrixHolder(
            embed_dim // num_heads,
            num_vectors=num_vectors,
            num_matrices=num_matrices,
            redraw_on_call=True,
        )
        print(f"matrix_holder head dim: {matrix_holder.head_dim}")
        super().__init__(
            *args,
            embed_dim=embed_dim,
            num_heads=num_heads,
            block_fn=partial(RFABlock, matrix_holder=matrix_holder),
            **kwargs,
        )


@register_model
def rfa_vit_tiny_patch16(pretrained=False, img_size=224, **kwargs):
    sizes = vit_sizes["Ti"]
    model = RFAViT(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **{**sizes, **kwargs},
    )
    return model


@register_model
def rfa_vit_small_patch16(pretrained=False, img_size=224, **kwargs):
    sizes = vit_sizes["S"]
    model = RFAViT(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **{**sizes, **kwargs},
    )
    return model


@register_model
def rfa_lra(input_dim, max_seq_len, **kwargs):
    sizes = dict(embed_dim=512, depth=4, num_heads=8, mlp_ratio=2.0)
    model = RFATransformer(
        input_dim=input_dim, max_seq_len=max_seq_len, **{**sizes, **kwargs}
    )
    return model


@register_model
def rfa_lra_imdb(input_dim, max_seq_len, **kwargs):
    sizes = dict(embed_dim=256, depth=4, num_heads=4, mlp_ratio=4.0)
    model = RFATransformer(
        input_dim=input_dim, max_seq_len=max_seq_len, **{**sizes, **kwargs}
    )
    return model


@register_model
def rfa_lra_cifar(input_dim, max_seq_len, **kwargs):
    sizes = dict(embed_dim=256, depth=1, num_heads=4, mlp_ratio=1.0)
    model = RFATransformer(
        input_dim=input_dim, max_seq_len=max_seq_len, **{**sizes, **kwargs}
    )
    return model
