from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from ..data_processing.skeleton import forward_kinematics
from . import geometry as geo

@dataclass
class FlowMatchingConfig:
    sigma_dir: float = 0.9
    fk_metric_alpha: float = 0.0
    source_coupling: str = "iid"
    laplacian_alpha: float = 0.5
    time_smoothing: float = 0.0
    time_smoothing_pinned: bool = False
    source_mode: str = "white"
    common_scale: float = 0.0
    sigma_ramp_kind: str = "linear"

class GraphFlowMatching(nn.Module):

    def __init__(self, field: nn.Module, cfg: FlowMatchingConfig, skeleton):
        super().__init__()
        self.field = field
        self.cfg = cfg
        self.skeleton = skeleton
        self.register_buffer("smoother", self._build_smoother(skeleton, cfg),
                             persistent=False)
        self.register_buffer("parents",
                             torch.as_tensor(skeleton.parents, dtype=torch.long),
                             persistent=False)
        self._ramp_cache: dict = {}

    @staticmethod
    def _build_smoother(skeleton, cfg) -> torch.Tensor:
        a = skeleton.adjacency()
        lap = np.diag(a.sum(1)) - a
        cov = np.linalg.inv(np.eye(len(a)) + cfg.laplacian_alpha * lap)
        w, v = np.linalg.eigh(cov)
        root = v @ np.diag(np.sqrt(np.clip(w, 0.0, None))) @ v.T

        root = root / np.sqrt(np.diag(root @ root.T))[:, None]
        return torch.from_numpy(root.astype(np.float32))

    def _ar1_filter(self, horizon, device, dtype):
        key = (horizon, str(device), str(dtype), float(self.cfg.time_smoothing),
               bool(self.cfg.time_smoothing_pinned))
        if getattr(self, "_ar1_cache_key", None) != key:
            rho = float(self.cfg.time_smoothing)
            idx = torch.arange(horizon, device=device, dtype=torch.float64)
            lag = idx[:, None] - idx[None, :]
            m = torch.where(lag >= 0, rho ** lag.clamp(min=0), torch.zeros_like(lag))
            if self.cfg.time_smoothing_pinned:

                m *= (1.0 - rho * rho) ** 0.5
            else:

                m[:, 1:] *= (1.0 - rho * rho) ** 0.5
            self._ar1_cache_key = key
            self._ar1_cache = m.to(dtype)
        return self._ar1_cache

    def zero_velocity_extrapolation(self, past, horizon):
        return past[:, -1:].expand(-1, horizon, -1, -1).contiguous()

    def _sigma_ramp(self, horizon, device, dtype):
        cfg = self.cfg
        key = (horizon, str(device), str(dtype), cfg.sigma_ramp_kind)
        if key not in self._ramp_cache:
            u = torch.arange(horizon, dtype=torch.float64) / max(horizon - 1, 1)
            if cfg.sigma_ramp_kind == "constant":
                ramp = torch.ones_like(u)
            elif cfg.sigma_ramp_kind == "linear":
                ramp = u
            elif cfg.sigma_ramp_kind == "sqrt":
                ramp = u.sqrt()
            else:
                raise ValueError(f"unknown sigma_ramp_kind {cfg.sigma_ramp_kind!r}")
            ramp = ramp / ramp[-1].clamp_min(1e-8)
            self._ramp_cache[key] = ramp.to(device=device, dtype=dtype)
        return self._ramp_cache[key]

    def sample_source(self, past, horizon, generator=None):
        cfg = self.cfg
        center = self.zero_velocity_extrapolation(past, horizon)
        xi = torch.randn(center.shape, device=center.device, dtype=center.dtype,
                         generator=generator)

        if cfg.source_mode not in ("white", "common_plus_white"):
            raise ValueError(f"unknown source_mode {cfg.source_mode!r}")
        if cfg.source_mode == "common_plus_white" and cfg.common_scale != 0.0:

            shape = (center.shape[0], 1) + tuple(center.shape[2:])
            zeta = torch.randn(shape, device=center.device, dtype=center.dtype,
                               generator=generator)
            ramp = self._sigma_ramp(horizon, center.device, center.dtype)
            xi = xi + cfg.common_scale * ramp[None, :, None, None] * zeta

        if cfg.source_coupling == "laplacian":
            xi = torch.einsum("ij,btjc->btic", self.smoother.to(xi.dtype), xi)
        elif cfg.source_coupling != "iid":
            raise ValueError(f"unknown source_coupling {cfg.source_coupling!r}")

        if cfg.time_smoothing > 0.0:
            xi = torch.einsum("tq,bqjc->btjc",
                              self._ar1_filter(xi.shape[1], xi.device, xi.dtype), xi)

        dirs = geo.sphere_exp(
            center[..., 1:, :],
            cfg.sigma_dir * geo.project_tangent(center[..., 1:, :], xi[..., 1:, :]),
        )
        return torch.cat([center[..., :1, :], dirs], dim=-2)

    def sample_time(self, batch, device, dtype):
        return torch.sigmoid(torch.randn(batch, device=device, dtype=dtype))

    def training_pair(self, source, target, s):
        s_b = s.view(-1, 1, 1, 1)
        return geo.product_geodesic(source, target, s_b)

    def forward(self, past, future, bone_lengths, generator=None):
        return self.loss(past, future, bone_lengths, generator=generator)

    def loss(self, past, future, bone_lengths, generator=None):
        cfg = self.cfg
        horizon = future.shape[1]
        source = self.sample_source(past, horizon, generator=generator)
        s = self.sample_time(future.shape[0], future.device, future.dtype)
        x_s, u_star = self.training_pair(source, future, s)

        dropped = geo.product_cut_locus_mask(source, future)
        keep = (~dropped).to(future.dtype)

        pred = self.field(x_s, past, s, bone_lengths)

        err = (pred - u_star).pow(2).sum(-1)
        loss_dir = (err[..., 1:] * keep).sum() / keep.sum().clamp_min(1.0)

        loss_fk = torch.zeros((), device=err.device, dtype=err.dtype)
        dir_term = loss_dir
        if cfg.fk_metric_alpha > 0.0:
            resid = pred - u_star
            resid = torch.cat([torch.zeros_like(resid[..., :1, :]),
                               resid[..., 1:, :] * keep.unsqueeze(-1)], dim=-2)
            lengths_e = bone_lengths[:, None, :].expand(resid.shape[:-1])

            err_fk = forward_kinematics(resid, lengths_e, self.parents)

            scale = bone_lengths[..., 1:].pow(2).mean().clamp_min(1e-8)
            loss_fk = err_fk[..., 1:, :].pow(2).sum(-1).mean() / scale
            a = cfg.fk_metric_alpha
            dir_term = (1.0 - a) * loss_dir + a * loss_fk

        total = dir_term

        stats = {
            "loss": total.detach(),
            "loss_dir": loss_dir.detach(),
            "loss_fk": loss_fk.detach(),
            "target_norm": u_star[..., 1:, :].norm(dim=-1).mean().detach(),
            "pred_norm": pred[..., 1:, :].norm(dim=-1).mean().detach(),
            "dropped_frac": dropped.to(err.dtype).mean().detach(),
        }
        return total, stats

    @torch.no_grad()
    def _integrate(self, x, past, bone_lengths, num_steps, solver,
                   return_trajectory):
        eta = 1.0 / num_steps
        traj = [x] if return_trajectory else None

        for n in range(num_steps):
            s_n = torch.full((x.shape[0],), n * eta, device=x.device, dtype=x.dtype)
            v = self.field(x, past, s_n, bone_lengths)
            if solver == "euler":
                x = geo.product_exp(x, eta * v)
            elif solver == "midpoint":
                x_mid = geo.product_exp(x, 0.5 * eta * v)
                v_mid = self.field(x_mid, past, s_n + 0.5 * eta, bone_lengths)

                v_mid = geo.product_project_tangent(
                    x, geo.product_transport(x_mid, x, v_mid))
                x = geo.product_exp(x, eta * v_mid)
            elif solver == "heun":
                x_hat = geo.product_exp(x, eta * v)
                v_hat = self.field(x_hat, past, s_n + eta, bone_lengths)
                v_hat = geo.product_project_tangent(
                    x, geo.product_transport(x_hat, x, v_hat))
                x = geo.product_exp(x, 0.5 * eta * (v + v_hat))
            else:
                raise ValueError(f"unknown solver {solver!r}")
            x = geo.renormalize_state(x)
            if return_trajectory:
                traj.append(x)

        return (x, torch.stack(traj, dim=1)) if return_trajectory else (x, None)

    @torch.no_grad()
    def sample(self, past, bone_lengths, horizon, num_steps: int = 20,
               num_samples: int = 1, generator=None, solver: str = "euler",
               return_trajectory: bool = False, chunk_size: int = 256):
        b = past.shape[0]
        if num_samples > 1:
            past = past.repeat_interleave(num_samples, dim=0)
            bone_lengths = bone_lengths.repeat_interleave(num_samples, dim=0)

        source = self.sample_source(past, horizon, generator=generator)

        total = past.shape[0]
        step = total if chunk_size <= 0 else min(chunk_size, total)
        outs, trajs = [], []
        for start in range(0, total, step):
            stop = min(start + step, total)
            x, traj = self._integrate(source[start:stop], past[start:stop],
                                      bone_lengths[start:stop], num_steps,
                                      solver, return_trajectory)
            outs.append(x)
            if return_trajectory:
                trajs.append(traj)

        x = torch.cat(outs, dim=0)
        out = x.view(b, num_samples, horizon, x.shape[-2], 3)
        if return_trajectory:
            stacked = torch.cat(trajs, dim=0)
            return out, stacked.view(b, num_samples, num_steps + 1, horizon,
                                     x.shape[-2], 3)
        return out
