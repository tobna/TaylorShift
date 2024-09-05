import logging
from functools import partial
import torch
from torch import nn
from timm.models.vision_transformer import Block
from vit import TimmViT
from utils import list_flatten
from math import sqrt


@torch.jit.script
def box_tensor(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Calculate a ⊠ b.

    Args:
        a (torch.Tensor): Tensor of shape (..., N, d)
        b (torch.Tensor): Tensor of shape (..., N, d)

    Returns:
        torch.Tensor: Tensor of shape (..., N, d^2)
    """
    return (a.unsqueeze(-1) * b.unsqueeze(-2)).view(list(a.shape)[:-1] + [-1])


class TaylorShiftAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        head_dim=None,
        qkv_bias=True,
        proj_drop=0.0,
        attn_drop=0.0,
        a=0.5,
        b=1.0,
        c=1.0,
        normalize_input=True,
        output_normalized=False,
    ):
        super().__init__()
        self.head_dim = int(dim / num_heads) if head_dim is None else head_dim
        self.qkv = nn.Linear(dim, num_heads * self.head_dim * 3, bias=qkv_bias)
        self.num_heads = num_heads
        self.proj = nn.Linear(num_heads * self.head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.scale = self.head_dim**0.25 if normalize_input else 1.0
        self.normalize_input = normalize_input
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
        if normalize_input:
            self.temperature = nn.Parameter(torch.ones(1, num_heads, 1, 1))

    def extra_repr(self) -> str:
        return f"num_heads={self.num_heads}, head_dim={self.head_dim}; a={self.a}, b={self.b}, c={self.c}; N0={self.N0}"

    def _efficient_attention(self, q, k, v):
        B, H, N, d = q.shape
        if self.output_normalized:
            v = torch.cat([sqrt(d / N) * torch.ones(*v.shape[:-1], 1, device=v.device, dtype=v.dtype), v], dim=-1)
        else:
            v = torch.cat([torch.ones(*v.shape[:-1], 1, device=v.device, dtype=v.dtype), v], dim=-1)
        if self.normalize_input:
            q = torch.nn.functional.normalize(q, dim=-1) * (self.scale * self.temperature)
            k = torch.nn.functional.normalize(k, dim=-1) * self.scale
            v = v / N

        kv_mod = box_tensor(k, k).transpose(-1, -2) @ v
        kv_mod = self.attn_drop(kv_mod)

        y = (
            self.a * box_tensor(q, q) @ kv_mod
            + (self.scale**2 * self.b) * q @ (k.transpose(-1, -2) @ v)
            + (self.scale**4 * self.c) * v.sum(-2).view(B, H, 1, d + 1)
        )

        y_norm, y = y[:, :, :, :1], y[:, :, :, 1:]
        return y / y_norm  # B x H x N x d

    def _direct_attention(self, q, k, v):
        if self.normalize_input:
            B, H, N, d = q.shape
            q = torch.nn.functional.normalize(q, dim=-1) * self.temperature
            k = torch.nn.functional.normalize(k, dim=-1)

        attn = q @ k.transpose(-2, -1)
        attn_max = attn.abs().max(dim=-1).values.view(*attn.shape[:-1], 1)
        max_val = self.a * attn_max.square() + self.b * attn_max + self.c
        if self.output_normalized:
            max_val *= sqrt(d / N)
        attn = self.a * (attn / max_val.sqrt()).square() + self.b * (attn / max_val) + self.c / max_val

        attn = attn / attn.sum(-1).view(*attn.shape[:-1], 1)
        attn = self.attn_drop(attn)
        return attn @ v

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )  # 3 x B x H x N x d
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
        q = q * (int(C) ** (-0.5))

        if N > self.N0:
            x = self._efficient_attention(q, k, v)
        else:
            x = self._direct_attention(q, k, v)

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class TaylorShiftBlock(Block):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        a=0.5,
        b=1.0,
        c=1.0,
        normalize_input=True,
        output_normalized=False,
        **kwargs,
    ):
        super().__init__(dim, num_heads, proj_drop=proj_drop, attn_drop=attn_drop, qkv_bias=qkv_bias, **kwargs)
        self.attn = TaylorShiftAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            normalize_input=normalize_input,
            output_normalized=output_normalized,
        )

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


class TaylorShiftViT(TimmViT):
    def __init__(
        self,
        img_size=224,
        a=0.5,
        b=1.0,
        c=1.0,
        normalize_input=True,
        output_normalized=True,
        conv_embed=False,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        conv_norm=nn.SyncBatchNorm,
        **kwargs,
    ):
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            block_fn=partial(
                TaylorShiftBlock, a=a, b=b, c=c, normalize_input=normalize_input, output_normalized=output_normalized
            ),
            **kwargs,
        )
        logging.info(f"adapted N0 = {self.blocks[0].attn.N0}")
        if conv_embed:
            from xcit import ConvPatchEmbed

            self.patch_embed = ConvPatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, conv_norm=conv_norm
            )
            self.embed_layer = partial(ConvPatchEmbed, conv_norm=conv_norm)

    @property
    def N0(self):
        return self.blocks[0].attn.N0

    @N0.setter
    def N0(self, N0):
        for block in self.blocks:
            block.attn.N0 = N0
        logging.info(f"set N0 = {self.blocks[0].attn.N0}")


class TextConvEmbed(nn.Module):
    def __init__(
        self, input_dim, embed_dim=768, bias=False, conv_norm=lambda channels: nn.GroupNorm(1, channels), depth=3
    ):
        super().__init__()
        self.proj = nn.Sequential(
            *list_flatten(
                [
                    [
                        nn.Conv1d(
                            input_dim if i == 0 else embed_dim // 2 ** (depth - i),
                            embed_dim // 2 ** (depth - i - 1),
                            3,
                            padding=1,
                            bias=bias,
                        ),
                        conv_norm(embed_dim // 2 ** (depth - i - 1)),
                        nn.GELU(),
                    ]
                    for i in range(depth)
                ]
            )
        )

    def forward(self, x):
        x = x.transpose(-1, -2)
        x = self.proj(x)
        x = x.transpose(-1, -2)
        return x


class TaylorShiftTransformer(TaylorShiftViT):
    def __init__(self, max_seq_len, input_dim, conv_embed=False, normalize_input=True, **kwargs):
        super().__init__(normalize_input=normalize_input, **kwargs)
        self.patch_embed = (
            TextConvEmbed(input_dim, self.embed_dim) if conv_embed else nn.Linear(input_dim, self.embed_dim)
        )
        pos_embed = torch.zeros(max_seq_len, self.embed_dim)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / self.embed_dim)
        )

        pos_embed[:, 0::2] = torch.sin(pos * div_term)
        pos_embed[:, 1::2] = torch.cos(pos * div_term)

        self.register_buffer("pe", pos_embed.view(1, max_seq_len, self.embed_dim))
        del self.pos_embed

    def set_max_seq_len(self, max_seq_len):
        pos_embed = torch.zeros(max_seq_len, self.embed_dim)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / self.embed_dim)
        )

        pos_embed[:, 0::2] = torch.sin(pos * div_term)
        pos_embed[:, 1::2] = torch.cos(pos * div_term)
        self.pe = pos_embed.view(1, max_seq_len, self.embed_dim)

    def _pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pe[:, : x.size(1)]
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pe[:, : x.size(1)]
        return self.pos_drop(x)

    def set_image_res(self, res):
        return

    def init_weights(self, mode=""):
        if self.cls_token is not None:
            nn.init.normal_(self.cls_token, std=1e-6)
