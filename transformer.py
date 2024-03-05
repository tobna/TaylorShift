import torch
from timm.models import register_model
from torch import nn
from vit import TimmViT
from utils import vit_sizes


class Transformer(TimmViT):
    def __init__(self, max_seq_len, input_dim, **kwargs):
        super().__init__(**kwargs)
        self.patch_embed = nn.Linear(input_dim, self.embed_dim)
        pos_embed = torch.zeros(max_seq_len, self.embed_dim)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / self.embed_dim))

        pos_embed[:, 0::2] = torch.sin(pos * div_term)
        pos_embed[:, 1::2] = torch.cos(pos * div_term)

        self.register_buffer('pe', pos_embed.view(1, max_seq_len, self.embed_dim))
        del self.pos_embed

    def set_max_seq_len(self, max_seq_len):
        pos_embed = torch.zeros(max_seq_len, self.embed_dim)
        pos = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / self.embed_dim))

        pos_embed[:, 0::2] = torch.sin(pos * div_term)
        pos_embed[:, 1::2] = torch.cos(pos * div_term)
        self.pe = pos_embed.view(1, max_seq_len, self.embed_dim)

    def forward(self, x, debug=False):
        if len(x.shape) >= 4 and x.shape[0] == 1:
            x = x.squeeze(0)
        return super().forward(x, debug=debug)

    def _pos_embed(self, x):
        if self.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + self.pe[:, :x.size(1)]
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if self.cls_token is not None:
                x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.pe[:, :x.size(1)]
        return self.pos_drop(x)

    def set_image_res(self, res):
        return


@register_model
def transformer_classifier_ti(input_dim, max_seq_len, **kwargs):
    sizes = vit_sizes['Ti']
    return Transformer(max_seq_len=max_seq_len, input_dim=input_dim, **{**sizes, **kwargs})

@register_model
def transformer_classifier_s(input_dim, max_seq_len, **kwargs):
    sizes = vit_sizes['S']
    return Transformer(max_seq_len=max_seq_len, input_dim=input_dim, **{**sizes, **kwargs})

@register_model
def transformer_classifier_b(input_dim, max_seq_len, **kwargs):
    sizes = vit_sizes['B']
    return Transformer(max_seq_len=max_seq_len, input_dim=input_dim, **{**sizes, **kwargs})

@register_model
def transformer_classifier_l(input_dim, max_seq_len, **kwargs):
    sizes = vit_sizes['L']
    return Transformer(max_seq_len=max_seq_len, input_dim=input_dim, **{**sizes, **kwargs})

@register_model
def transformer_lra(input_dim, max_seq_len, **kwargs):
    sizes = dict(embed_dim=512, depth=4, num_heads=8, mlp_ratio=2.)
    return Transformer(max_seq_len=max_seq_len, input_dim=input_dim, **{**sizes, **kwargs})

@register_model
def transformer_lra_imdb(input_dim, max_seq_len, **kwargs):
    sizes = dict(embed_dim=256, depth=4, num_heads=4, mlp_ratio=4.)
    return Transformer(max_seq_len=max_seq_len, input_dim=input_dim, **{**sizes, **kwargs})

@register_model
def transformer_lra_cifar(input_dim, max_seq_len, **kwargs):
    sizes = dict(embed_dim=256, depth=1, num_heads=4, mlp_ratio=1.)
    return Transformer(max_seq_len=max_seq_len, input_dim=input_dim, **{**sizes, **kwargs})

@register_model
def transformer_lra_path(input_dim, max_seq_len, **kwargs):
    sizes = dict(embed_dim=64, depth=4, num_heads=4, mlp_ratio=1.)
    return Transformer(max_seq_len=max_seq_len, input_dim=input_dim, **{**sizes, **kwargs})
