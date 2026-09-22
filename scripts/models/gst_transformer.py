from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data_processing.skeleton import Skeleton, forward_kinematics
from .embeddings import MLP, SinusoidalEncoding, modulate
from .geometry import product_project_tangent



@dataclass
class GSTConfig:

    hidden: int = 384
    num_layers: int = 8
    num_heads: int = 8
    ffn_mult: float = 4.0
    dropout: float = 0.0
    time_embed_dim: int = 128

    spatial_mode: str = "attention"
    spatial_hops: int = 0

    temporal_mode: str = "attention"
    temporal_window: int = 0

    max_hop_bucket: int = 11

    tie_layers: bool = False
    adaln_mode: str = "per_block"

    use_positions: bool = True
    use_velocity_feature: bool = True
    frame_period: float = 1.0 / 60.0

    def head_dim(self) -> int:
        if self.hidden % self.num_heads:
            raise ValueError("hidden must be divisible by num_heads")
        return self.hidden // self.num_heads

GST_PRESETS: dict[str, dict] = {
    "base": dict(hidden=384, num_layers=8, num_heads=8, time_embed_dim=128),
    "small_nospatial": dict(hidden=256, num_layers=4, num_heads=8, time_embed_dim=96,
                            spatial_mode="none"),
    "small_fixed": dict(hidden=256, num_layers=4, num_heads=8, time_embed_dim=96,
                        spatial_mode="fixed"),
    "small_1hop": dict(hidden=256, num_layers=4, num_heads=8, time_embed_dim=96,
                       spatial_hops=1),
    "small_nospatial_nofk": dict(hidden=256, num_layers=4, num_heads=8,
                                 time_embed_dim=96, spatial_mode="none",
                                 use_positions=False, use_velocity_feature=False),
    "small_full": dict(hidden=256, num_layers=4, num_heads=8, time_embed_dim=96),
    "small_notemporal": dict(hidden=256, num_layers=4, num_heads=8,
                             time_embed_dim=96, temporal_mode="none"),
    "small_notemporal_novel": dict(hidden=256, num_layers=4, num_heads=8,
                                   time_embed_dim=96, temporal_mode="none",
                                   use_velocity_feature=False),
    "small_temporal_1hop": dict(hidden=256, num_layers=4, num_heads=8,
                                time_embed_dim=96, temporal_window=1),
    "small_adaln_single": dict(hidden=256, num_layers=4, num_heads=8,
                               time_embed_dim=96, adaln_mode="single"),
    "small_adaln_single_deep": dict(hidden=256, num_layers=6, num_heads=8,
                                    time_embed_dim=96, adaln_mode="single"),
    "base_adaln_single_deep": dict(hidden=384, num_layers=12, num_heads=8,
                                   time_embed_dim=128, adaln_mode="single"),
    "base_tied_deep": dict(hidden=384, num_layers=12, num_heads=8,
                           time_embed_dim=128, tie_layers=True, adaln_mode="single"),
}


class Attention(nn.Module):

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, bias=None):
        n, length, _ = x.shape
        qkv = self.qkv(x).reshape(n, length, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        if bias is not None:
            bias = bias.to(q.dtype)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=bias,
        )
        out = out.transpose(1, 2).reshape(n, length, -1)
        return self.proj(out)


class FixedGraphMix(nn.Module):

    def __init__(self, dim: int, mix: torch.Tensor):
        super().__init__()
        self.register_buffer("mix", mix, persistent=False)
        self.proj = nn.Linear(dim, dim)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, bias=None):
        return self.proj(torch.einsum("ji,nic->njc", self.mix.to(x.dtype), x))


class Identity0(nn.Module):

    def forward(self, x, bias=None):
        return torch.zeros_like(x)


class GSTBlock(nn.Module):

    def __init__(self, cfg: GSTConfig, spatial_mix: torch.Tensor | None = None):
        super().__init__()
        h = cfg.hidden
        self.norm_sp = nn.LayerNorm(h, elementwise_affine=False, eps=1e-6)
        self.norm_tm = nn.LayerNorm(h, elementwise_affine=False, eps=1e-6)
        self.norm_ff = nn.LayerNorm(h, elementwise_affine=False, eps=1e-6)

        if cfg.spatial_mode == "attention":
            self.attn_sp = Attention(h, cfg.num_heads)
        elif cfg.spatial_mode == "fixed":
            self.attn_sp = FixedGraphMix(h, spatial_mix)
        elif cfg.spatial_mode == "none":
            self.attn_sp = Identity0()
        else:
            raise ValueError(f"unknown spatial_mode {cfg.spatial_mode!r}")

        if cfg.temporal_mode == "attention":
            self.attn_tm = Attention(h, cfg.num_heads)
        elif cfg.temporal_mode == "none":
            self.attn_tm = Identity0()
        else:
            raise ValueError(f"unknown temporal_mode {cfg.temporal_mode!r}")

        inner = int(h * cfg.ffn_mult)
        self.ffn = nn.Sequential(
            nn.Linear(h, inner), nn.GELU(approximate="tanh"),
            nn.Dropout(cfg.dropout), nn.Linear(inner, h),
        )

        if cfg.adaln_mode == "per_block":
            self.ada = nn.Sequential(nn.SiLU(), nn.Linear(h, 9 * h))
            nn.init.zeros_(self.ada[-1].weight)
            nn.init.zeros_(self.ada[-1].bias)
            self.ada_offset = None
        elif cfg.adaln_mode == "single":
            self.ada = None
            self.ada_offset = nn.Parameter(torch.zeros(9 * h))
        else:
            raise ValueError(f"unknown adaln_mode {cfg.adaln_mode!r}")

    def forward(self, h, cond, spatial_bias, temporal_bias, shapes, mod=None):
        b, t, j, c = shapes
        if self.ada is not None:
            raw = self.ada(cond)
        else:
            if mod is None:
                raise ValueError("adaln_mode='single' needs the shared modulation")
            raw = mod + self.ada_offset
        mods = raw.chunk(9, dim=-1)
        sp_shift, sp_scale, sp_gate, tm_shift, tm_scale, tm_gate, \
            ff_shift, ff_scale, ff_gate = [m[:, None, None, :] for m in mods]

        y = modulate(self.norm_sp(h), sp_shift, sp_scale)
        y = self.attn_sp(y.reshape(b * t, j, c), spatial_bias).reshape(b, t, j, c)
        h = h + sp_gate * y

        y = modulate(self.norm_tm(h), tm_shift, tm_scale)
        y = y.permute(0, 2, 1, 3).reshape(b * j, t, c)
        y = self.attn_tm(y, temporal_bias)
        y = y.reshape(b, j, t, c).permute(0, 2, 1, 3)
        h = h + tm_gate * y

        y = modulate(self.norm_ff(h), ff_shift, ff_scale)
        h = h + ff_gate * self.ffn(y)
        return h


class GSTVelocityField(nn.Module):

    def __init__(self, cfg: GSTConfig, skeleton: Skeleton):
        super().__init__()
        self.cfg = cfg
        self.skeleton = skeleton
        h, d = cfg.hidden, cfg.time_embed_dim
        j = skeleton.num_joints

        hop = torch.from_numpy(skeleton.hop_distance()).clamp(max=cfg.max_hop_bucket)
        self.register_buffer("hop", hop, persistent=False)
        self.register_buffer("relation",
                             torch.from_numpy(skeleton.relation()), persistent=False)
        self.register_buffer("parents", torch.from_numpy(skeleton.parents_array()),
                             persistent=False)
        self.register_buffer("struct",
                             torch.from_numpy(skeleton.structural_features()),
                             persistent=False)

        self.enc_frame = SinusoidalEncoding(d, max_period=8.0, min_period=0.01)
        self.enc_gen = SinusoidalEncoding(d, max_period=2.0, min_period=0.01)
        self.enc_offset = SinusoidalEncoding(d, max_period=8.0, min_period=0.01)

        in_dim = 3 + d + 1 + 1
        if cfg.use_positions:
            in_dim += 3
        if cfg.use_velocity_feature:
            in_dim += 3
        in_dim += self.struct.shape[-1] + d
        self.phi_in = MLP(in_dim, h, h)

        self.cond = nn.Sequential(nn.Linear(d, h), nn.SiLU(), nn.Linear(h, h))

        self.hop_bias = nn.Embedding(cfg.max_hop_bucket + 1, cfg.num_heads)
        self.rel_bias = nn.Embedding(5, cfg.num_heads)
        nn.init.zeros_(self.hop_bias.weight)
        nn.init.zeros_(self.rel_bias.weight)
        self.time_bias = MLP(d, h // 2, cfg.num_heads)

        mix = None
        if cfg.spatial_mode == "fixed":
            a = torch.from_numpy(skeleton.adjacency()) + torch.eye(j)
            mix = a / a.sum(dim=1, keepdim=True).clamp_min(1.0)
        if cfg.tie_layers:

            self.block = GSTBlock(cfg, mix)
            self.blocks = None

            self.iter_embed = nn.Parameter(torch.zeros(cfg.num_layers, h))
            nn.init.normal_(self.iter_embed, std=0.02)
        else:
            self.block = None
            self.blocks = nn.ModuleList(
                [GSTBlock(cfg, mix) for _ in range(cfg.num_layers)])
            self.iter_embed = None

        if cfg.adaln_mode == "single":
            self.ada_shared = nn.Sequential(nn.SiLU(), nn.Linear(h, 9 * h))
            nn.init.zeros_(self.ada_shared[-1].weight)
            nn.init.zeros_(self.ada_shared[-1].bias)
        else:
            self.ada_shared = None
        self.norm_out = nn.LayerNorm(h, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(h, 2 * h))
        self.phi_out = nn.Linear(h, 3)
        nn.init.zeros_(self.ada_out[-1].weight)
        nn.init.zeros_(self.ada_out[-1].bias)
        nn.init.zeros_(self.phi_out.weight)
        nn.init.zeros_(self.phi_out.bias)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _spatial_bias(self, device, dtype):
        bias = self.hop_bias(self.hop) + self.rel_bias(self.relation)
        bias = bias.permute(2, 0, 1).unsqueeze(0)
        if self.cfg.spatial_hops > 0:
            mask = self.hop > self.cfg.spatial_hops
            bias = bias.masked_fill(mask[None, None], float("-inf"))
        return bias.to(dtype)

    def _temporal_bias(self, num_frames, device, dtype):
        idx = torch.arange(num_frames, device=device)
        offset = (idx[None, :] - idx[:, None]).to(dtype) * self.cfg.frame_period
        bias = self.time_bias(self.enc_offset(offset))
        bias = bias.permute(2, 0, 1).unsqueeze(0)
        if self.cfg.temporal_window > 0:
            mask = (idx[None, :] - idx[:, None]).abs() > self.cfg.temporal_window
            bias = bias.masked_fill(mask[None, None], float("-inf"))
        return bias.to(dtype)

    def forward(self, state, past, s, bone_lengths):
        cfg = self.cfg
        b, tf, j, _ = state.shape
        tp = past.shape[1]
        t = tp + tf
        device, dtype = state.device, state.dtype

        x = torch.cat([past, state], dim=1)

        lengths = bone_lengths[:, None, :].expand(b, t, j)
        pos = forward_kinematics(x, lengths, self.parents)

        frame_idx = torch.arange(-tp + 1, tf + 1, device=device, dtype=dtype)
        e_t = self.enc_frame(frame_idx * cfg.frame_period)
        e_t = e_t[None, :, None, :].expand(b, t, j, -1)
        beta = torch.zeros(t, device=device, dtype=dtype)
        beta[:tp] = 1.0
        beta = beta[None, :, None, None].expand(b, t, j, 1)
        e_s = self.enc_gen(s)

        feats = [x, e_t, beta, lengths.unsqueeze(-1),
                 e_s[:, None, None, :].expand(b, t, j, -1)]
        if cfg.use_positions:
            feats.append(pos)
        if cfg.use_velocity_feature:
            vel = torch.zeros_like(pos)
            vel[:, 1:] = pos[:, 1:] - pos[:, :-1]
            feats.append(vel)
        feats.append(self.struct.to(dtype)[None, None].expand(b, t, j, -1))

        h = self.phi_in(torch.cat(feats, dim=-1))

        cond = self.cond(e_s)
        spatial_bias = (self._spatial_bias(device, h.dtype)
                        if cfg.spatial_mode == "attention" else None)
        temporal_bias = (self._temporal_bias(t, device, h.dtype)
                         if cfg.temporal_mode == "attention" else None)

        shapes = (b, t, j, cfg.hidden)
        shared = self.ada_shared
        if self.blocks is not None:
            mod = shared(cond) if shared is not None else None
            for block in self.blocks:
                h = block(h, cond, spatial_bias, temporal_bias, shapes, mod)
        else:
            for k in range(cfg.num_layers):

                cond_k = cond + self.iter_embed[k]
                mod = shared(cond_k) if shared is not None else None
                h = self.block(h, cond_k, spatial_bias, temporal_bias, shapes, mod)

        shift, scale = self.ada_out(cond).chunk(2, dim=-1)
        h = modulate(self.norm_out(h), shift[:, None, None], scale[:, None, None])

        ambient = self.phi_out(h[:, tp:])
        velocity = product_project_tangent(state, ambient)
        return torch.cat(
            [torch.zeros_like(velocity[..., :1, :]), velocity[..., 1:, :]], dim=-2)


def build_velocity_field(preset: str, skeleton: Skeleton, **overrides) -> GSTVelocityField:
    if preset not in GST_PRESETS:
        raise KeyError(f"unknown preset {preset!r}; choose from {sorted(GST_PRESETS)}")
    cfg = GSTConfig(**{**GST_PRESETS[preset], **overrides})
    return GSTVelocityField(cfg, skeleton)
