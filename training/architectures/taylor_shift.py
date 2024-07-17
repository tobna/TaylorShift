import torch
from timm.models import register_model
from torch import nn
from architectures.taylor_shift_vit import TaylorSoftmaxViT, TaylorSoftmaxAttentionNoNorm


def _list_flatten(listlist):
    return [item for sublist in listlist for item in sublist]


class TextConvEmbed(nn.Module):
    def __init__(
        self,
        input_dim,
        embed_dim=768,
        bias=False,
        conv_norm=lambda channels: nn.GroupNorm(1, channels),
        depth=3,
    ):
        super().__init__()
        self.proj = nn.Sequential(
            *_list_flatten(
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


class TaylorTransformer(TaylorSoftmaxViT):
    def __init__(self, max_seq_len, input_dim, conv_embed=False, xcit_version=True, **kwargs):
        super().__init__(xcit_version=xcit_version, **kwargs)
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


class TaylorTransformerNoNorm(TaylorTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for block in self.blocks:
            TaylorSoftmaxAttentionNoNorm.cast(block.attn)


@register_model
def taylor_shift_triv_convembed_listops(input_dim, max_seq_len, **kwargs):
    sizes = dict(embed_dim=512, depth=4, num_heads=8, mlp_ratio=2.0)
    model = TaylorTransformer(
        max_seq_len=max_seq_len,
        input_dim=input_dim,
        **{**sizes, **kwargs},
        conv_embed=True,
        output_normalized=True,
    )
    model.N0 = 1_000_000
    return model


@register_model
def taylor_shift_triv_listops(input_dim, max_seq_len, **kwargs):
    sizes = dict(embed_dim=512, depth=4, num_heads=8, mlp_ratio=2.0)
    model = TaylorTransformer(
        max_seq_len=max_seq_len,
        input_dim=input_dim,
        **{**sizes, **kwargs},
        conv_embed=False,
        output_normalized=True,
    )
    model.N0 = 1_000_000
    return model


@register_model
def taylor_shift_eff_listops(input_dim, max_seq_len, **kwargs):
    sizes = dict(embed_dim=512, depth=4, num_heads=8, mlp_ratio=2.0)
    model = TaylorTransformer(
        max_seq_len=max_seq_len,
        input_dim=input_dim,
        **{**sizes, **kwargs},
        conv_embed=False,
        output_normalized=True,
    )
    model.N0 = 0
    return model


@register_model
def taylor_shift_triv_cifar(input_dim, max_seq_len, **kwargs):
    sizes = dict(embed_dim=256, depth=1, num_heads=4, mlp_ratio=1.0)
    model = TaylorTransformer(
        max_seq_len=max_seq_len,
        input_dim=input_dim,
        **{**sizes, **kwargs},
        conv_embed=False,
        output_normalized=True,
    )
    model.N0 = 1_000_000
    return model


@register_model
def taylor_shift_triv_imdb(input_dim, max_seq_len, **kwargs):
    sizes = dict(embed_dim=256, depth=4, num_heads=4, mlp_ratio=4.0)
    model = TaylorTransformer(
        max_seq_len=max_seq_len,
        input_dim=input_dim,
        **{**sizes, **kwargs},
        conv_embed=False,
        output_normalized=True,
    )
    model.N0 = 1_000_000
    return model


@register_model
def taylor_shift_eff_imdb(input_dim, max_seq_len, **kwargs):
    sizes = dict(embed_dim=256, depth=4, num_heads=4, mlp_ratio=4.0)
    model = TaylorTransformer(
        max_seq_len=max_seq_len,
        input_dim=input_dim,
        **{**sizes, **kwargs},
        conv_embed=False,
        output_normalized=True,
    )
    model.N0 = 0
    return model


@register_model
def taylor_shift_eff_cifar(input_dim, max_seq_len, **kwargs):
    sizes = dict(embed_dim=256, depth=1, num_heads=4, mlp_ratio=1.0)
    model = TaylorTransformer(
        max_seq_len=max_seq_len,
        input_dim=input_dim,
        **{**sizes, **kwargs},
        conv_embed=False,
        output_normalized=True,
    )
    model.N0 = 0
    return model
