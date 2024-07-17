# adapted from https://github.com/HKUNLP/efficient-attention/blob/main/efficient-attention/efficient_attention/eva.py
# with slight modifications

import math
import warnings
from functools import partial

import numpy as np
from typing import Optional
import torch
from timm.models import register_model
from timm.models.vision_transformer import Block
from torch import nn
from timm.models.layers import trunc_normal_
from torch import Tensor
from torch.nn import functional as F
from einops import rearrange

from architectures.transformer import Transformer
from architectures.vit import TimmViT
from resizing_interface import vit_sizes


class MultiheadAttention(nn.Module):
    def __init__(
        self, dim, num_heads, fp32=False, qkv_bias=True, attn_drop=0.0, proj_drop=0.0
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv_bias = qkv_bias
        self.fp32 = fp32

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def proj_and_split_heads(self, x):
        B, *seq_shape, C = x.shape
        N = np.prod(seq_shape)
        # x now has shape [b, n, c]
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        return q, k, v

    def forward(self, x, key_padding_mask=None):
        B, *seq_shape, C = x.shape
        q, k, v = self.proj_and_split_heads(x)

        output = self._apply_attention(q, k, v, key_padding_mask)

        x = output.transpose(1, 2).reshape((B,) + tuple(seq_shape) + (C,))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _apply_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        key_padding_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ):
        r"""
        Computes scaled dot product attention on query, key and value tensors, using
        an optional attention mask if passed, and applying dropout if a probability
        greater than 0.0 is specified.
        Returns a tensor pair containing attended values and attention weights.
        Args:
            q, k, v: query, key and value tensors. See Shape section for shape details.
            attn_mask: optional tensor containing mask values to be added to calculated
                attention. May be 2D or 3D; see Shape section for details.
        Shape:
            - q: :math:`(B, Nt, E)` where B is batch size, Nt is the target sequence length,
                and E is embedding dimension.
            - key: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
                and E is embedding dimension.
            - value: :math:`(B, Ns, E)` where B is batch size, Ns is the source sequence length,
                and E is embedding dimension.
            - attn_mask: either a 3D tensor of shape :math:`(B, Nt, Ns)` or a 2D tensor of
                shape :math:`(Nt, Ns)`.
            - Output: attention values have shape :math:`(B, Nt, E)`; attention weights
                have shape :math:`(B, Nt, Ns)`
        """
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            attn = attn + attn_mask  # attn_weights = attn_weights + attn_mask

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        output = attn @ v
        return output


def default(val, d):
    return val if val is not None else d


def nonoverlap_window_2d_partition(x, window_size):
    """
    Args:
        x: (b, h, H, W, d)
        window_size (int): window size
    Returns:
        windows: (num_windows * num_windows, window_size * window_size, C)
    """
    *_, H, W, d = x.shape
    num_windows_h = H // window_size
    num_windows_w = W // window_size
    output = rearrange(
        x,
        "... (h1 h) (w1 w) d -> ... (h1 w1) (h w) d",
        h1=num_windows_h,
        w1=num_windows_w,
        h=window_size,
        w=window_size,
    )
    return output


def nonoverlap_window_1d_partition(x, window_size):
    return rearrange(x, "... (g w) d -> ... g w d", w=window_size)


def window_1d_partition(x, window_size, ext_window_size=0, pad_val=0):
    b, h, n, d = x.shape
    n_groups = n // window_size
    if ext_window_size > 0:
        ext_len = ext_window_size
        x = F.pad(x, (0, 0, ext_len, ext_len), value=pad_val)
        out_shape = (b, h, n_groups, 2 * ext_len + window_size, d)
        strides = x.stride()
        out_stride = (
            strides[0],
            strides[1],
            window_size * strides[2],
            strides[2],
            strides[3],
        )
        return torch.as_strided(x, size=out_shape, stride=out_stride)
    else:
        return nonoverlap_window_1d_partition(x, window_size)


def window_1d_merge(x):
    return rearrange(x, "... g w d ->... (g w) d")


def window_2d_partition(x, window_size, ext_window_size=0, pad_val=0):
    """
    Args:
        x: (b, h, H, W, d)
        window_size (int): Window size
    Returns:
        x: (b, h, num_groups, group_size, d)
    """
    if ext_window_size > 0:
        b, h, H, W, d = x.shape
        total_window_size = 2 * ext_window_size + window_size
        x = F.pad(
            x,
            [0, 0, ext_window_size, ext_window_size, ext_window_size, ext_window_size],
            value=pad_val,
        )
        out_shape = [
            b,
            h,
            H // window_size,
            W // window_size,
            total_window_size,
            total_window_size,
            d,
        ]
        in_stride = x.stride()
        out_stride = [
            in_stride[0],
            in_stride[1],
            in_stride[2] * window_size,
            in_stride[3] * window_size,
            in_stride[2],
            in_stride[3],
            in_stride[4],
        ]
        output = x.as_strided(size=out_shape, stride=out_stride)
        return rearrange(output, "... h1 w1 h w d -> ... (h1 w1) (h w) d")
    else:
        return nonoverlap_window_2d_partition(x, window_size)


def window_2d_merge(x, window_size, hw_tuple):
    """
    Args:
        x: (b, h, num_windows * num_windows, window_size * window_size, d)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (b, h, H, W, d)
    """
    assert isinstance(hw_tuple, (list, tuple))
    H, W = hw_tuple
    b, h, num_windows_sq, window_size_sq, d = x.shape
    assert window_size**2 == window_size_sq
    num_windows_h = H // window_size
    num_windows_w = W // window_size
    assert num_windows_sq == num_windows_h * num_windows_w
    output = rearrange(
        x,
        "... (h1 w1) (h w) d -> ... (h1 h) (w1 w) d",
        h1=num_windows_h,
        w1=num_windows_w,
        h=window_size,
        w=window_size,
    )
    return output


# adapted from https://github.com/lucidrains/local-attention/blob/master/local_attention/local_attention.py
def pad_to_multiple(tensor, multiple, dim=-2, value=0, create_mask=False):
    assert dim < 0  # only accept ``dim'' index in a reverse manner
    seqlen = int(tensor.shape[dim])
    multiple = int(multiple)
    m = seqlen / multiple
    if m == int(m):
        if create_mask:
            return tensor, torch.zeros(
                size=(tensor.shape[0], tensor.shape[-2]),
                dtype=torch.bool,
                device=tensor.device,
            )
        else:
            return tensor
    remainder = math.ceil(m) * multiple - seqlen
    pad_offset = (0,) * (-1 - dim) * 2
    padded_res = F.pad(tensor, (*pad_offset, 0, remainder), value=value)
    if create_mask:
        # assume dim 0 is the batch size
        padding_mask = torch.zeros(
            size=(padded_res.shape[0], padded_res.shape[-2]),
            dtype=torch.bool,
            device=padded_res.device,
        )
        padding_mask[:, -remainder:] = True
        return padded_res, padding_mask
    else:
        return padded_res


# adapted from https://github.com/NVIDIA/transformer-ls/blob/master/lra/attention_transformer_ls.py
class LocalAttention(MultiheadAttention):

    def __init__(
        self,
        use_rpe=False,
        window_size=2,
        attn_2d=False,
        overlap_window=False,
        *args,
        **kwargs
    ):
        super(LocalAttention, self).__init__(*args, **kwargs)
        self.window_size = window_size
        self.attn_2d = attn_2d
        self.use_rpe = use_rpe if window_size > 0 else False
        if overlap_window:
            self.ext_size = max(1, self.window_size // 2)
        else:
            self.ext_size = 0
        if self.use_rpe:
            if attn_2d:
                # handle the boarder conditions...
                w_pad = self.ext_size
                self.local_relative_position_bias_table = nn.Parameter(
                    torch.zeros(
                        2 * (window_size + w_pad - 1) * (2 * w_pad + window_size + 1)
                        + 1,
                        self.num_heads,
                    )
                )
                trunc_normal_(self.local_relative_position_bias_table, std=0.02)

                # get pair-wise relative position index
                coords_h = torch.arange(-w_pad, w_pad + window_size)
                coords_w = torch.arange(-w_pad, w_pad + window_size)
                coords = torch.stack(
                    torch.meshgrid([coords_h, coords_w], indexing="ij")
                )  # 2, 2w, 2w
                coords = (
                    coords.view(2, (window_size + w_pad * 2) ** 2)
                    .transpose(0, 1)
                    .unsqueeze(0)
                )  # 1, 4w**2, 2
                q_coords_hw = torch.arange(0, window_size)
                q_coords = torch.stack(
                    torch.meshgrid([q_coords_hw, q_coords_hw])
                )  # 2, w, w
                q_coords = (
                    q_coords.view(2, window_size**2).transpose(0, 1).unsqueeze(1)
                )  # w**2, 1, 2
                relative_coords = q_coords - coords
                relative_coords += w_pad + window_size - 1  # shift to start from 0
                relative_coords[:, :, 0] *= 2 * w_pad + window_size
                relative_position_index = relative_coords.sum(-1)  # w^2, 4w^2
                self.register_buffer("relative_position_index", relative_position_index)
            else:
                self.local_relative_position_bias_table = nn.Parameter(
                    torch.zeros(
                        self.num_heads, window_size, window_size + self.ext_size * 2
                    )
                )
                trunc_normal_(self.local_relative_position_bias_table, std=0.02)
        self.apply(self._init_weights)

    def add_rel_pos_bias(self, local_dots):
        if self.attn_2d:
            local_relative_position_bias = self.local_relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(
                1,
                self.window_size * self.window_size,
                (self.ext_size * 2 + self.window_size) ** 2,
                -1,
            )
            local_relative_position_bias = local_relative_position_bias.permute(
                0, 3, 1, 2
            ).unsqueeze(2)
        else:
            local_relative_position_bias = (
                self.local_relative_position_bias_table.unsqueeze(0).unsqueeze(2)
            )
        return local_dots + local_relative_position_bias

    def window_partition(self, x, shape, ext_window_size, pad_val=0, window_size=None):
        window_size = default(window_size, self.window_size)
        if self.attn_2d:
            assert isinstance(shape, (list, tuple))
            H, W = shape
            return window_2d_partition(
                rearrange(x, "... (H W) d ->... H W d", H=H, W=W),
                window_size=window_size,
                ext_window_size=ext_window_size,
                pad_val=pad_val,
            )
        else:
            return window_1d_partition(
                x,
                window_size=window_size,
                ext_window_size=ext_window_size,
                pad_val=pad_val,
            )

    def window_merge(self, x, shape, window_size=None):
        window_size = default(window_size, self.window_size)
        if self.attn_2d:
            assert isinstance(shape, (list, tuple))
            output = window_2d_merge(x, window_size=window_size, hw_tuple=shape)
            return rearrange(output, "... H W d ->... (H W) d")
        else:
            return window_1d_merge(x)

    def _process_input(self, x, key_padding_mask):
        # this function is used in its children attention classes.
        B, *seq_shape, C = x.shape
        N = np.prod(seq_shape)
        if self.attn_2d:
            assert len(seq_shape) == 2
            if self.window_size > 0:
                assert (
                    seq_shape[0] % self.window_size == 0
                    and seq_shape[1] % self.window_size == 0
                )
            x = x.reshape(B, N, C)
        else:
            if self.window_size > 0:
                if key_padding_mask is None:
                    x, key_padding_mask = pad_to_multiple(
                        x, self.window_size, dim=-2, create_mask=True
                    )
                else:
                    x = pad_to_multiple(x, self.window_size, dim=-2)
                    key_padding_mask = pad_to_multiple(
                        key_padding_mask, self.window_size, dim=-1, value=True
                    )
                N = x.shape[-2]
                seq_shape = [N]
        return x, key_padding_mask, seq_shape

    def _apply_attention(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        key_padding_mask: Optional[Tensor] = None,
    ):
        mask_val = -5e4
        if self.attn_2d:
            b, h, n, d = q.shape
            H = W = int(math.sqrt(n))
            shape = (H, W)
            assert H * W == n
            orig_n = n
        else:
            orig_n = q.shape[-2]
            if key_padding_mask is None:
                q, key_padding_mask = pad_to_multiple(
                    q, self.window_size, dim=-2, create_mask=True
                )
            else:
                q = pad_to_multiple(q, self.window_size, dim=-2)
                key_padding_mask = pad_to_multiple(
                    key_padding_mask, self.window_size, dim=-1, value=True
                )
            k, v = map(lambda t: pad_to_multiple(t, self.window_size, dim=-2), (k, v))
            b, h, n, d = q.shape
            shape = n
        if key_padding_mask is None:
            key_padding_mask = torch.zeros(b, n, dtype=q.dtype, device=q.device)
        key_padding_mask = (
            key_padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool)
        )  # [b, 1, n, 1]

        w_q = self.window_partition(q, shape, ext_window_size=0)
        w_k = self.window_partition(k, shape, ext_window_size=self.ext_size)
        w_v = self.window_partition(v, shape, ext_window_size=self.ext_size)
        local_dots = (
            torch.einsum("bhwie,bhwje->bhwij", w_q, w_k) * self.scale
        )  # [b, h, w, i, j]

        if self.use_rpe:
            local_dots = self.add_rel_pos_bias(local_dots)

        local_dots_mask = (
            self.window_partition(
                key_padding_mask, shape, ext_window_size=self.ext_size, pad_val=1
            )
            .to(torch.bool)
            .transpose(-1, -2)
        )
        local_dots.masked_fill_(local_dots_mask, mask_val)

        local_attn = local_dots.softmax(dim=-1)
        output = torch.einsum("bhwij,bhwje->bhwie", local_attn, w_v)

        output = self.window_merge(output, shape)[..., :orig_n, :]
        return output


def prm_projection(
    data: torch.Tensor,
    projection_matrix: torch.Tensor,
    normalize: bool = True,
    diagonal: bool = False,
    return_exp: bool = False,
    is_query: bool = False,
    eps: float = 1e-8,
):
    """
    Constructs nonnegative kernel features for fast softmax attention.
    Args:
    data: input for which features are computes
    projection_matrix: random matrix used to compute features
    batch_dims_t: tuple of batch dimensions
    is_query: predicate indicating whether input data corresponds to queries or
        keys
    eps: numerical stabilizer.
    Returns:
    Random features for fast softmax attention.
    """
    # data : [n, b, h, lk, d]
    # proj : [n, b, h, lc, d]
    # We have e^{qk^T/sqrt{d}} = e^{q_norm k_norm^T}, where
    # w_norm = w * data_normalizer for w in {q,k}.
    # NOTE: scaler with 0.5 could considerably stablizes training.
    # now test norm also uses scaled data: that is, multiply by data.shape[-1] ** -1.
    # normalized_data = (data.shape[-1] ** -0.5) * data
    # data_dash = torch.einsum('...nd,...md->...nm',
    #                         projection_matrix,
    #                         normalized_data,
    #                         ) # [n, b, h, c, lq]
    # norm = torch.sum(normalized_data ** 2, dim=-1).unsqueeze(-2) / 2.0# [n, b, h, 1, lk]
    data_normalizer = data.shape[-1] ** -0.5
    if diagonal:
        data_dash = torch.einsum(
            "...nd,...nd->...n",
            projection_matrix,
            (data_normalizer * data),
        )  # [n, b, h, lq, lk]
        norm = data_normalizer * torch.sum(data**2, dim=-1) / 2.0  # [n, b, h, 1, lk]
    else:
        data_dash = torch.einsum(
            "...nd,...md->...nm",
            projection_matrix,
            (data_normalizer * data),
        )  # [n, b, h, lq, lk]
        norm = (
            data_normalizer * torch.sum(data**2, dim=-1).unsqueeze(-2) / 2.0
        )  # [n, b, h, 1, lk]
    if normalize:
        proj_data = F.softmax(data_dash - norm, dim=-1)  # [n, b, h, l_c, l_k]
    elif return_exp:
        if is_query:
            proj_data = (
                torch.exp(
                    data_dash
                    - norm
                    - torch.amax(data_dash, dim=-2, keepdim=True).detach()
                )
                + eps
            )
        else:
            proj_data = (
                torch.exp(
                    data_dash
                    - norm
                    - torch.amax(data_dash, dim=(-1, -2, -3), keepdim=True).detach()
                )
                + eps
            )
    else:
        proj_data = data_dash - norm
    return proj_data


class EVAAttn(LocalAttention):
    def __init__(
        self,
        adaptive_proj="default",
        num_landmarks=49,
        use_t5_rpe=False,
        *args,
        **kwargs
    ):
        super(EVAAttn, self).__init__(*args, **kwargs)
        self.adaptive_proj = adaptive_proj
        if self.adaptive_proj in ["default"]:
            self.adaptive_mu_q = nn.Sequential(
                nn.Linear(self.head_dim, self.head_dim),
                nn.LayerNorm(self.head_dim),
            )
            self.adaptive_mu_k = nn.Sequential(
                nn.Linear(self.head_dim, self.head_dim),
                nn.LayerNorm(self.head_dim),
            )
        elif self.adaptive_proj in ["no-ln"]:
            self.adaptive_mu_q = nn.Sequential(
                nn.Linear(self.head_dim, self.head_dim),
            )
            self.adaptive_mu_k = nn.Sequential(
                nn.Linear(self.head_dim, self.head_dim),
            )
        elif self.adaptive_proj in ["none"]:
            self.adaptive_mu_k = nn.Sequential(
                nn.Linear(self.head_dim, self.head_dim),
                nn.LayerNorm(self.head_dim),
            )
        self.use_t5_rpe = use_t5_rpe
        self.num_landmarks = num_landmarks
        if self.use_rpe and not self.use_t5_rpe:
            warnings.warn(
                "By setting --use-rpe, the default relative positional embedding for local window is used."
                "We also implement a T5-style positional encoding, which we observe performs slightly better;"
                "This feature can be enabled by only setting --use-t5-rpe."
            )
        elif self.use_rpe and self.use_t5_rpe:
            raise NotImplementedError(
                "Default RPE and T5-style RPE cannot be true simultaneously."
            )
        if self.use_t5_rpe:
            raise NotImplementedError("T5-style RPE is not implemented yet.")
            # self.rel_pos_bias = T5RelativePositionBias(
            #     self.scale,
            #     num_heads = self.num_heads,
            #     causal = False,
            #     num_buckets=max(min(int((self.window_size + self.ext_size) / 2), 64), 16),
            #     max_distance=self.window_size + self.ext_size
            # )
        self.apply(self._init_weights)

    def _process_input(self, x, key_padding_mask):
        # this function re-implements the parent method.
        B, *seq_shape, C = x.shape
        N = np.prod(seq_shape)
        if self.attn_2d:
            assert len(seq_shape) == 2
            if self.window_size > 0:
                assert (
                    seq_shape[0] % self.window_size == 0
                    and seq_shape[1] % self.window_size == 0
                )
        else:
            if self.window_size > 0:
                if key_padding_mask is None:
                    x, key_padding_mask = pad_to_multiple(
                        x, self.window_size, dim=-2, create_mask=True
                    )
                else:
                    x = pad_to_multiple(x, self.window_size, dim=-2)
                    key_padding_mask = pad_to_multiple(
                        key_padding_mask, self.window_size, dim=-1, value=True
                    )
                N = x.shape[-2]
                seq_shape = [N]
        return x, key_padding_mask, seq_shape

    def forward(self, x, key_padding_mask=None):
        mask_val = -5e4
        ######################## Generate Proposal Parameters ###############################
        B, *seq_shape, C = x.shape
        orig_n = np.prod(seq_shape)
        x, key_padding_mask, seq_shape = self._process_input(x, key_padding_mask)
        N = np.prod(seq_shape)
        q, k, v = self.proj_and_split_heads(x)

        if key_padding_mask is None:
            key_padding_mask = torch.zeros(B, N, dtype=k.dtype, device=k.device)
        key_padding_mask = (
            key_padding_mask.unsqueeze(1).unsqueeze(-1).to(torch.bool)
        )  # [b, 1, n, 1]

        w_q = self.window_partition(q, seq_shape, ext_window_size=0)
        w_k = self.window_partition(k, seq_shape, ext_window_size=self.ext_size)
        w_v = self.window_partition(
            v, seq_shape, ext_window_size=self.ext_size
        )  # [b, h, w, j, d]

        if self.attn_2d:
            rf_win_size = int(math.sqrt(N // self.num_landmarks))
        else:
            rf_win_size = int(N // self.num_landmarks)
        # [b, h, c, j, d]
        rf_w_q = self.window_partition(
            q, seq_shape, window_size=rf_win_size, ext_window_size=self.ext_size
        )
        # [b, h, c, j, d]
        rf_w_k = self.window_partition(
            k, seq_shape, window_size=rf_win_size, ext_window_size=self.ext_size
        )
        # [b, h, c, j, d]
        rf_w_v = self.window_partition(
            v, seq_shape, window_size=rf_win_size, ext_window_size=self.ext_size
        )
        # compute local attention
        # [b, 1, c, j, 1]
        rf_w_mask = self.window_partition(
            key_padding_mask,
            seq_shape,
            window_size=rf_win_size,
            ext_window_size=self.ext_size,
            pad_val=1,
        ).to(torch.bool)
        rf_w_q = rf_w_q.masked_fill(rf_w_mask, 0.0)
        rf_w_k = rf_w_k.masked_fill(rf_w_mask, 0.0)
        rf_w_v = rf_w_v.masked_fill(rf_w_mask, 0.0)

        if self.adaptive_proj in ["default", "no-ln"]:
            rf_q_bar = self.adaptive_mu_q(rf_w_q.mean(dim=-2))
            rf_k_bar = self.adaptive_mu_k(rf_w_k.mean(dim=-2))
            # [b, h, c, d]
            mu = 0.5 * (rf_q_bar + rf_k_bar)
        elif self.adaptive_proj == "none":
            rf_k_bar = self.adaptive_mu_k(rf_w_k.mean(dim=-2))
            mu = torch.zeros_like(rf_k_bar)
        ######################## Sampling from proposal ###############################
        if self.training:
            weights = mu + torch.randn_like(mu)
        else:
            weights = mu
            # [b, h, c, j, d], [b, h, c, 1, d] -> [b, h, c, j]
        log_proj_w_k = prm_projection(
            rf_w_k, weights.unsqueeze(-2), normalize=False
        ).squeeze(-2)
        log_proj_w_k = log_proj_w_k.masked_fill(rf_w_mask.squeeze(-1), mask_val)

        # [b, h, c, j] [b, h, c, j, d] -> [b, h, c, d]
        beta = torch.einsum(
            "...cj,...cjd->...cd", torch.softmax(log_proj_w_k, dim=-1), rf_w_v
        )

        # compute approx. expectation of CVs.
        # [b, h, c, d]
        rfa_chunk = torch.einsum("...wid,...cd->...wic", w_q, self.scale * rf_k_bar)
        num_rfa_chunks = rfa_chunk.shape[-1]

        # compute local attention
        local_dots_mask = (
            self.window_partition(
                key_padding_mask, seq_shape, ext_window_size=self.ext_size, pad_val=1
            )
            .to(torch.bool)
            .transpose(-1, -2)
        )

        log_qk_local_dot = (
            torch.einsum("bhwie,bhwje->bhwij", w_q, w_k) * self.scale
        )  # [b, h, w, i, j]
        if self.use_t5_rpe:
            # here the t5-rpe-bias has already been scaled by \sqrt{d}
            log_qk_local_dot = log_qk_local_dot + self.rel_pos_bias(log_qk_local_dot)
        if self.use_rpe:
            log_qk_local_dot = self.add_rel_pos_bias(log_qk_local_dot)

        log_qk_local_dot = log_qk_local_dot.masked_fill(local_dots_mask, mask_val)
        local_len = log_qk_local_dot.shape[-1]

        # compute attention weights along with normalizing constant.
        attn = torch.softmax(torch.cat([log_qk_local_dot, rfa_chunk], dim=-1), dim=-1)
        local_attn, ra_attn = torch.split(attn, [local_len, num_rfa_chunks], dim=-1)
        output_local = torch.einsum("bhwij,bhwjd->bhwid", local_attn, w_v)
        output_snis = torch.einsum("bhwic,bhcd->bhwid", ra_attn, beta)
        ######################## Combine them together ############################
        output = self.window_merge(
            output_snis + output_local, seq_shape
        )  # [b, h, n, d]
        x = output.permute(0, 2, 1, 3).reshape((B,) + tuple(seq_shape) + (C,))
        x = self.proj(x)
        if orig_n is not None:
            x = x[..., :orig_n, :]
        x = self.proj_drop(x)
        return x


class EVABlock(Block):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=False,
        qk_norm=False,
        attn_drop=0.0,
        proj_drop=0.0,
        **kwargs
    ):
        super().__init__(dim, num_heads, **kwargs)
        assert not qk_norm, "qk_norm is not supported in EVA block."
        eva_args = dict(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            adaptive_proj="no-ln",  # use default or no-ln?
            num_landmarks=49,
            use_t5_rpe=False,
            use_rpe=False,
            window_size=8,
            attn_2d=False,  # we want the model to learn the 2D structure, just like all other models also have to.
            overlap_window=True,
        )
        self.attn = EVAAttn(**eva_args)


class EVAViT(TimmViT):
    def __init__(self, **kwargs):
        super().__init__(block_fn=EVABlock, **kwargs)


class EVA(Transformer):
    def __init__(self, **kwargs):
        super().__init__(block_fn=EVABlock, **kwargs)


@register_model
def eva_vit_tiny_patch16(pretrained=False, img_size=224, **kwargs):
    sizes = vit_sizes["Ti"]
    model = EVAViT(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **{**sizes, **kwargs}
    )
    return model


@register_model
def eva_vit_small_patch16(pretrained=False, img_size=224, **kwargs):
    sizes = vit_sizes["S"]
    model = EVAViT(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **{**sizes, **kwargs}
    )
    return model


@register_model
def eva_lra(input_dim, max_seq_len, **kwargs):
    sizes = dict(embed_dim=512, depth=4, num_heads=8, mlp_ratio=2.0)
    return EVA(max_seq_len=max_seq_len, input_dim=input_dim, **{**sizes, **kwargs})


@register_model
def eva_lra_imdb(input_dim, max_seq_len, **kwargs):
    sizes = dict(embed_dim=256, depth=4, num_heads=4, mlp_ratio=4.0)
    return EVA(max_seq_len=max_seq_len, input_dim=input_dim, **{**sizes, **kwargs})


@register_model
def eva_lra_cifar(input_dim, max_seq_len, **kwargs):
    sizes = dict(embed_dim=256, depth=1, num_heads=4, mlp_ratio=1.0)
    return EVA(max_seq_len=max_seq_len, input_dim=input_dim, **{**sizes, **kwargs})
