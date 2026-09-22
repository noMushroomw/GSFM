from __future__ import annotations

import torch

_ANGLE_EPS = 1e-6

_COS_EPS = 1e-6

_REJECT_EPS = 1e-6


def _safe_norm(x, dim=-1, keepdim=True, eps=1e-12):
    return torch.sqrt(x.pow(2).sum(dim=dim, keepdim=keepdim).clamp_min(eps))


def project_tangent(p, w):
    return w - (p * w).sum(-1, keepdim=True) * p


def sphere_exp(p, w):
    norm = _safe_norm(w)
    small = norm < _ANGLE_EPS
    sinc = torch.where(small, torch.ones_like(norm), torch.sin(norm) / norm)
    out = torch.cos(norm) * p + sinc * w
    return out / _safe_norm(out)


def sphere_log(p, q):
    inner = (p * q).sum(-1, keepdim=True)
    rho = torch.arccos(inner.clamp(-1.0 + _COS_EPS, 1.0 - _COS_EPS))
    v = q - inner * p
    out = rho * v / _safe_norm(v)
    raw_norm = v.pow(2).sum(-1, keepdim=True).sqrt()
    return torch.where(raw_norm < _REJECT_EPS, torch.zeros_like(out), out)


def is_cut_locus(p, q, tol: float = 1e-4):
    return (p * q).sum(-1) < -1.0 + tol


def sphere_transport(p, q, w):
    inner = (p * q).sum(-1, keepdim=True)
    denom = 1.0 + inner
    safe = denom.abs() > _REJECT_EPS
    out = w - (q * w).sum(-1, keepdim=True) / torch.where(
        safe, denom, torch.ones_like(denom)) * (p + q)
    return torch.where(safe, out, project_tangent(q, w))


def geodesic_point_and_velocity(z, y, s):
    inner = (z * y).sum(-1, keepdim=True)
    rho = torch.arccos(inner.clamp(-1.0 + _COS_EPS, 1.0 - _COS_EPS))
    v = y - inner * z
    a = v / _safe_norm(v)
    raw_norm = v.pow(2).sum(-1, keepdim=True).sqrt()

    srho = s * rho
    d_s = torch.cos(srho) * z + torch.sin(srho) * a
    u_s = rho * (-torch.sin(srho) * z + torch.cos(srho) * a)

    degenerate = raw_norm < _REJECT_EPS
    d_s = torch.where(degenerate, z, d_s)
    u_s = torch.where(degenerate, torch.zeros_like(u_s), u_s)

    d_s = d_s / _safe_norm(d_s)
    return d_s, u_s


def euclidean_point_and_velocity(z, y, s):
    return (1.0 - s) * z + s * y, y - z


def wrapped_normal(center, sigma):
    xi = torch.randn_like(center)
    return sphere_exp(center, sigma * project_tangent(center, xi))


def product_project_tangent(state, ambient):
    root = ambient[..., :1, :]
    bones = project_tangent(state[..., 1:, :], ambient[..., 1:, :])
    return torch.cat([root, bones], dim=-2)


def product_exp(state, tangent):
    root = state[..., :1, :] + tangent[..., :1, :]
    bones = sphere_exp(state[..., 1:, :], tangent[..., 1:, :])
    return torch.cat([root, bones], dim=-2)


def product_transport(p, q, tangent):
    root = tangent[..., :1, :]
    bones = sphere_transport(p[..., 1:, :], q[..., 1:, :], tangent[..., 1:, :])
    return torch.cat([root, bones], dim=-2)


def product_geodesic(z, y, s):
    r_s, ur_s = euclidean_point_and_velocity(z[..., :1, :], y[..., :1, :], s)
    d_s, ud_s = geodesic_point_and_velocity(z[..., 1:, :], y[..., 1:, :], s)
    return torch.cat([r_s, d_s], dim=-2), torch.cat([ur_s, ud_s], dim=-2)


def product_cut_locus_mask(z, y, tol: float = 1e-4):
    return is_cut_locus(z[..., 1:, :], y[..., 1:, :], tol=tol)


def renormalize_state(state):
    root = state[..., :1, :]
    bones = state[..., 1:, :]
    return torch.cat([root, bones / _safe_norm(bones)], dim=-2)
