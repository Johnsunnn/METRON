import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """
    From: https://github.com/huggingface/pytorch-image-models
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    DropPath is dropping an entire sample from the batch while Dropout is dropping random values
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class MetaboliteInteractionPerception(nn.Module):
    def __init__(self, input_embedding_dim, n_channels, kernel=5, no_off=False) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.input_embedding_dim = input_embedding_dim
        self.offset_range_factor = kernel
        self.no_off = no_off
        self.scale = 1 ** -0.5
        self.scale_factor = 1 ** -0.5
        self.proj_q = nn.Conv1d(self.n_channels, self.n_channels, kernel_size=1, stride=1, padding=0)
        self.proj_k = nn.Conv1d(self.n_channels, self.n_channels, kernel_size=1, stride=1, padding=0)
        self.proj_v = nn.Conv1d(self.n_channels, self.n_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = nn.Linear(self.input_embedding_dim, self.input_embedding_dim)
        kernel_size = kernel
        self.stride = 1
        pad_size = kernel_size // 2 if kernel_size != self.stride else 0
        self.proj_offset = nn.Sequential(
            nn.Conv1d(1, 1, kernel_size=kernel_size, stride=self.stride, padding=pad_size),
            nn.Conv1d(1, 1, kernel_size=1, stride=self.stride, padding=0),
        )

    def forward(self, x):
        B, _, E = x.shape
        dtype, device = x.dtype, x.device
        q = self.proj_q(x)
        offset = self.proj_offset(q)
        offset = rearrange(offset, 'b 1 n -> b n')
        def grid_sample_1d(feats, grid, **kwargs):
            grid = rearrange(grid, '... -> ... 1 1')
            grid = F.pad(grid, (1, 0), value=0.)
            feats = rearrange(feats, '... -> ... 1')
            out = F.grid_sample(feats, grid, **kwargs)
            return rearrange(out, '... 1 -> ...')
        def normalize_grid(arange, out_dim=-1):
            n = arange.shape[out_dim]
            return 2.0 * arange / max(n - 1, 1) - 1.0

        if self.offset_range_factor >= 0 and not self.no_off:
            offset = offset.tanh().mul(self.offset_range_factor)

        if self.no_off:
            x_sampled = F.avg_pool1d(x, kernel_size=self.stride, stride=self.stride)
        else:
            grid = torch.arange(offset.shape[-1], device=device, dtype=dtype)
            vgrid = grid + offset
            vgrid_scaled = normalize_grid(vgrid, out_dim=-1)
            x_sampled = grid_sample_1d(x, vgrid_scaled, mode='bilinear', padding_mode='zeros', align_corners=False)

        k = self.proj_k(x_sampled).reshape(B, 1, E)
        v = self.proj_v(x_sampled).reshape(B, 1, E)

        q_for_attn = q.transpose(-1, -2)
        k_for_attn = k.transpose(-1, -2)
        v_for_attn = v.transpose(-1, -2)

        scaled_dot_prod = torch.einsum('b i d , b j d -> b i j', q_for_attn, k_for_attn) * self.scale_factor
        attention = torch.softmax(scaled_dot_prod, dim=-1)
        out_attn = torch.einsum('b i j , b j d -> b i d', attention, v_for_attn)
        out_permuted = out_attn.permute(0, 2, 1)
        output_final = self.proj_out(out_permuted)

        return output_final
