import logging
from functools import partial
import torch
from timm.models import register_model
from torch import nn
from timm.models.vision_transformer import Block

from architectures.vit import TimmViT
from resizing_interface import vit_sizes
from math import sqrt


@torch.jit.script
def box_tensor(a, b):
    # logging.debug(f"box_tensor: a : {a.norm(dim=-1).max()}, b : {b.norm(dim=-1).max()}")
    return (a.unsqueeze(-1) * b.unsqueeze(-2)).view(list(a.shape)[:-1] + [-1])


class TaylorSoftmaxAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        head_dim=None,
        qkv_bias=True,
        proj_drop=0.0,
        drop=None,
        attn_drop=0.0,
        a=0.5,
        b=1.0,
        c=1.0,
        xcit_version=True,
        output_normalized=False,
    ):
        super().__init__()
        if drop is not None:
            proj_drop = drop
        self.head_dim = int(dim / num_heads) if head_dim is None else head_dim
        self.qkv = nn.Linear(dim, num_heads * self.head_dim * 3, bias=qkv_bias)
        self.num_heads = num_heads
        self.proj = nn.Linear(num_heads * self.head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.scale = self.head_dim**0.25
        self.xcit_version = xcit_version
        self.output_normalized = output_normalized
        assert a > 0 and c >= b**2 / (4 * a), (
            f"Choose a, b, c, such that ax^2+bx+c >= 0 for all x => "
            f"a > 0 and c >= b^2/(4a), but got a={a}, b={b}, c={c}."
        )
        self.a = a
        self.b = b
        self.c = c
        self.N0 = int(self.head_dim**2 + self.head_dim + 1)
        logging.debug(f"Using linear attention V2 with a={a}, b={b}, c={c}")
        if xcit_version:
            self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

    def extra_repr(self) -> str:
        return f"num_heads={self.num_heads}, head_dim={self.head_dim}; a={self.a}, b={self.b}, c={self.c}; N0={self.N0}"

    def _linear_attention(self, q, k, v, debug=False):
        B, H, N, d = q.shape
        if self.output_normalized:
            v = torch.cat(
                [
                    sqrt(d / N)
                    * torch.ones(*v.shape[:-1], 1, device=v.device, dtype=v.dtype),
                    v,
                ],
                dim=-1,
            )
        else:
            v = torch.cat(
                [torch.ones(*v.shape[:-1], 1, device=v.device, dtype=v.dtype), v],
                dim=-1,
            )
        if self.xcit_version:
            q = torch.nn.functional.normalize(q, dim=-1) * (
                self.scale * self.temperature
            )  # * N ** -.5
            k = torch.nn.functional.normalize(k, dim=-1) * (self.scale)  # * (N ** .5))
            v = v / N

            kv_mod = box_tensor(k, k).transpose(-1, -2) @ v
            kv_mod = self.attn_drop(kv_mod)

            y = (
                self.a * box_tensor(q, q) @ kv_mod
                + (self.scale**2 * self.b) * q @ (k.transpose(-1, -2) @ v)
                + (self.scale**4 * self.c) * v.sum(-2).view(B, H, 1, d + 1)
            )
        else:
            q_norm, k_norm = q.norm(2, dim=-1).detach().view(B, H, N, 1), k.norm(
                2, dim=-1
            ).square().sum(dim=-1).detach().view(B, H, 1, 1)
            q_normed, k_normed = q / q_norm, k / k_norm

            y = (
                self.a
                * box_tensor(q, q_normed)
                @ (box_tensor(k_normed, k).transpose(-1, -2) @ v)
                + self.b * q_normed @ (k_normed.transpose(-1, -2) @ v)
                + self.c
                * v.sum(-2).view(B, H, 1, d + 1)
                / (q_norm * k_norm).view(B, H, N, 1)
            )

        y_norm, y = y[:, :, :, :1], y[:, :, :, 1:]
        return y / y_norm  # B x H x N x d

    def _squared_attention(self, q, k, v, debug=False):
        if self.xcit_version:
            B, H, N, d = q.shape
            q = torch.nn.functional.normalize(q, dim=-1) * self.temperature
            k = torch.nn.functional.normalize(k, dim=-1)

        attn = q @ k.transpose(-2, -1)
        attn_max = attn.abs().max(dim=-1).values.view(*attn.shape[:-1], 1)
        max_val = self.a * attn_max.square() + self.b * attn_max + self.c
        attn = (
            self.a * (attn / max_val.sqrt()).square()
            + self.b * (attn / max_val)
            + self.c / max_val
        )
        attn = attn / attn.sum(-1).view(*attn.shape[:-1], 1)

        if self.output_normalized:
            # attn *= sqrt(N / d)
            v = sqrt(N / d) * v

        attn = self.attn_drop(attn)
        return attn @ v

    def forward(self, x, debug=False):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )  # 3 x B x H x N x d
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        if debug:
            if N > self.N0:
                x = self._linear_attention(q, k, v, debug=debug)
            else:
                x = self._squared_attention(q, k, v, debug=debug)
        else:
            if N > self.N0:
                x = self._linear_attention(q, k, v)
            else:
                x = self._squared_attention(q, k, v)
        if debug:
            logging.debug(f"direct attention output: {x.norm(dim=-1).mean()}")
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        if debug:
            logging.debug(f"attention block output: {x.norm(dim=-1).mean()}")
        return x


class TaylorSoftmaxAttentionNoNorm(TaylorSoftmaxAttention):
    @classmethod
    def cast(cls, attn: TaylorSoftmaxAttention):
        assert isinstance(
            attn, TaylorSoftmaxAttention
        ), f"Attention must be of type {TaylorSoftmaxAttention}, but is {type(attn)}."
        attn.__class__ = cls
        if attn.xcit_version:
            del attn.temperature
        assert isinstance(attn, TaylorSoftmaxAttentionNoNorm)

    def _linear_attention(self, q, k, v, debug=False):
        B, H, N, d = q.shape
        v = torch.cat(
            [torch.ones(*v.shape[:-1], 1, device=v.device, dtype=v.dtype), v], dim=-1
        )
        kv_mod = box_tensor(k, k).transpose(-1, -2) @ v
        kv_mod = self.attn_drop(kv_mod)

        y = (
            self.a * box_tensor(q, q) @ kv_mod
            + self.b * q @ (k.transpose(-1, -2) @ v)
            + self.c * v.sum(-2).view(B, H, 1, d + 1)
        )

        y_norm, y = y[:, :, :, :1], y[:, :, :, 1:]
        if debug:
            logging.debug(f"y_norm: mean -> {y_norm.mean()} -> min = {y_norm.min()}")
        return y / y_norm  # B x H x N x d

    def _squared_attention(self, q, k, v, debug=False):
        attn = q @ k.transpose(-2, -1)
        attn = self.a * attn.square() + self.b * attn + self.c

        attn = attn / attn.sum(-1).view(*attn.shape[:-1], 1)
        attn = self.attn_drop(attn)
        return attn @ v


class TaylorSoftmaxBlock(Block):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        drop=None,
        a=0.5,
        b=1.0,
        c=1.0,
        xcit_version=True,
        output_normalized=False,
        **kwargs,
    ):
        if drop is not None:
            proj_drop = drop
        try:
            super().__init__(
                dim,
                num_heads,
                proj_drop=proj_drop,
                attn_drop=attn_drop,
                qkv_bias=qkv_bias,
                **kwargs,
            )
        except TypeError:
            if proj_drop == 0:
                super().__init__(
                    dim, num_heads, attn_drop=attn_drop, qkv_bias=qkv_bias, **kwargs
                )
            else:
                super().__init__(
                    dim,
                    num_heads,
                    attn_drop=attn_drop,
                    qkv_bias=qkv_bias,
                    proj_drop=proj_drop,
                    **kwargs,
                )

        self.attn = TaylorSoftmaxAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            xcit_version=xcit_version,
            output_normalized=output_normalized,
        )

    def forward(self, x, debug=False):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), debug=debug)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class TaylorSoftmaxViT(TimmViT):
    def __init__(
        self,
        img_size=224,
        a=0.5,
        b=1.0,
        c=1.0,
        xcit_version=True,
        output_normalized=False,
        **kwargs,
    ):
        super().__init__(
            img_size=img_size,
            block_fn=partial(
                TaylorSoftmaxBlock,
                a=a,
                b=b,
                c=c,
                xcit_version=xcit_version,
                output_normalized=output_normalized,
            ),
            **kwargs,
        )
        logging.info(f"adapted N0 = {self.blocks[0].attn.N0}")

    @property
    def N0(self):
        return self.blocks[0].attn.N0

    @N0.setter
    def N0(self, N0):
        for block in self.blocks:
            block.attn.N0 = N0
        logging.info(f"set N0 = {self.blocks[0].attn.N0}")


class TaylorXCiT(TaylorSoftmaxViT):
    def __init__(
        self,
        *args,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        conv_norm=nn.SyncBatchNorm,
        **kwargs,
    ):
        super().__init__(
            *args,
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            **kwargs,
        )
        from architectures.xcit import ConvPatchEmbed

        self.patch_embed = ConvPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            conv_norm=conv_norm,
        )
        self.embed_layer = partial(ConvPatchEmbed, conv_norm=conv_norm)


@register_model
def taylor_shift_triv_vit_ti_p16(img_size=224, **kwargs):
    size = vit_sizes["Ti"]
    model = TaylorSoftmaxViT(
        img_size=img_size,
        in_chans=3,
        patch_size=16,
        xcit_version=True,
        conv_norm=lambda channels: nn.GroupNorm(1, channels),
        output_normalized=True,
        **{**size, **kwargs},
    )
    model.N0 = 1_000_000
    return model


@register_model
def taylor_shift_triv_vit_s_p16(img_size=224, **kwargs):
    size = vit_sizes["S"]
    model = TaylorSoftmaxViT(
        img_size=img_size,
        in_chans=3,
        patch_size=16,
        xcit_version=True,
        conv_norm=lambda channels: nn.GroupNorm(1, channels),
        output_normalized=True,
        **{**size, **kwargs},
    )
    model.N0 = 1_000_000
    return model


@register_model
def taylor_shift_eff_vit_s_p16(img_size=224, **kwargs):
    size = vit_sizes["S"]
    model = TaylorSoftmaxViT(
        img_size=img_size,
        in_chans=3,
        patch_size=16,
        xcit_version=True,
        conv_norm=lambda channels: nn.GroupNorm(1, channels),
        output_normalized=True,
        **{**size, **kwargs},
    )
    model.N0 = -1
    return model


@register_model
def taylor_shift_eff_vit_ti_p16(img_size=224, **kwargs):
    size = vit_sizes["Ti"]
    model = TaylorSoftmaxViT(
        img_size=img_size,
        in_chans=3,
        patch_size=16,
        xcit_version=True,
        conv_norm=lambda channels: nn.GroupNorm(1, channels),
        output_normalized=True,
        **{**size, **kwargs},
    )
    model.N0 = -1
    return model


@register_model
def taylor_shift_triv_vit_convemb_s_p16(img_size=224, **kwargs):
    size = vit_sizes["S"]
    model = TaylorXCiT(
        img_size=img_size,
        in_chans=3,
        patch_size=16,
        xcit_version=True,
        conv_norm=lambda channels: nn.GroupNorm(1, channels),
        **{**size, **kwargs},
    )
    model.N0 = 1_000_000
    return model
