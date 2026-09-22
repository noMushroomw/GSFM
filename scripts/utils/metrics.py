from __future__ import annotations

import numpy as np
import torch

CM = 100.0


def _apply_mask(x, joint_mask):
    return x if joint_mask is None else x[..., joint_mask, :]


def ade_fde(pred, target, joint_mask=None):
    pred = _apply_mask(pred, joint_mask)
    target = _apply_mask(target, joint_mask)
    dist = (pred - target.unsqueeze(1)).norm(dim=-1)
    ade = dist.mean(dim=(-1, -2)).min(dim=1).values * CM
    fde = dist[:, :, -1].mean(dim=-1).min(dim=1).values * CM
    return ade, fde


def ade_fde_conventional(pred, target, joint_mask=None):
    pred = _apply_mask(pred, joint_mask)
    target = _apply_mask(target, joint_mask)
    diff = pred - target.unsqueeze(1)
    per_frame = diff.flatten(start_dim=-2).norm(dim=-1)
    ade = per_frame.mean(dim=-1).min(dim=1).values
    fde = per_frame[:, :, -1].min(dim=1).values
    return ade, fde


def apd_conventional(pred, joint_mask=None):
    pred = _apply_mask(pred, joint_mask)
    b, n = pred.shape[:2]
    if n < 2:
        return torch.zeros(b, device=pred.device)
    flat = pred.reshape(b, n, -1)
    return torch.cdist(flat, flat).sum(dim=(1, 2)) / (n * (n - 1))


def mean_angle_error(pred, target, parents, joint_mask=None):
    def bone_dirs(x):
        parent_pos = x[..., parents.clamp(min=0), :]
        d = x - parent_pos
        return d / d.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    dist = (pred - target.unsqueeze(1)).norm(dim=-1).mean(dim=(-1, -2))
    best = dist.argmin(dim=1)
    chosen = pred[torch.arange(pred.shape[0], device=pred.device), best]
    dp = bone_dirs(chosen)[..., 1:, :]
    dt = bone_dirs(target)[..., 1:, :]
    cos = (dp * dt).sum(-1).clamp(-1.0, 1.0)
    return torch.rad2deg(torch.arccos(cos)).mean(dim=(-1, -2))


def apd(pred, joint_mask=None):
    pred = _apply_mask(pred, joint_mask)
    b, n = pred.shape[:2]
    if n < 2:
        return torch.zeros(b, device=pred.device)
    flat = pred.reshape(b, n, -1)
    dist = torch.cdist(flat, flat)
    per_frame_joint = flat.shape[-1] // 3
    total = dist.sum(dim=(1, 2)) / (n * (n - 1))

    return total / np.sqrt(per_frame_joint) * CM

_SD_AMASS_LIMBS = (
    (1, 2), (1, 3), (2, 3),
    (3, 6), (6, 9), (9, 12), (12, 15),
    (9, 14), (14, 17), (17, 19), (19, 21),
    (9, 13), (13, 16), (16, 18), (18, 20),
    (2, 5), (5, 8), (8, 11),
    (1, 4), (4, 7), (7, 10),
)

_SD_AMASS_CHAINS = ((0, 2, 3, 4, 5, 6), (0, 3), (4, 7, 8, 9, 10),
                    (4, 11, 12, 13, 14), (0, 15, 16, 17), (18, 19, 20))

_SD_H36M_LIMBS = (
    (1, 4), (1, 7), (4, 7),
    (1, 2), (2, 3),
    (4, 5), (5, 6),
    (7, 8), (8, 9), (9, 10),
    (8, 11), (8, 14),
    (11, 12), (12, 13),
    (14, 15), (15, 16),
)
_SD_H36M_CHAINS = ((3, 4), (0, 2, 7, 8, 9), (1, 7, 10, 12, 13), (7, 11, 14, 15))


def sd_limb_tables(num_joints: int):
    if num_joints == 22:
        return _SD_AMASS_LIMBS, _SD_AMASS_CHAINS
    if num_joints == 17:
        return _SD_H36M_LIMBS, _SD_H36M_CHAINS
    raise ValueError(
        f"no SkeletonDiffusion limb table for {num_joints} joints; their MAE is "
        f"defined per skeleton and the chains have to be transcribed from "
        f"src/data/skeleton/kinematic/ in their repository")


def mae_joint_angle(pred, target, limbs=None, chains=None):
    limbs = torch.as_tensor(_SD_AMASS_LIMBS if limbs is None else limbs,
                            device=pred.device)
    chains = _SD_AMASS_CHAINS if chains is None else chains
    pairs = torch.as_tensor([[c[i], c[i + 1]] for c in chains
                             for i in range(len(c) - 1)], device=pred.device)

    def angles(x):
        v = x[..., limbs[:, 1], :] - x[..., limbs[:, 0], :]
        a, b = v[..., pairs[:, 0], :], v[..., pairs[:, 1], :]
        cos = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-7)
        return torch.arccos(cos.clamp(-1.0, 1.0))

    diff = (angles(pred) - angles(target.unsqueeze(1))).abs()
    return torch.rad2deg(diff.mean(-1).mean(-1)).min(dim=1).values


def apde(pred, mm_targets, joint_mask=None):
    out = []
    nan = torch.tensor(float("nan"), device=pred.device, dtype=pred.dtype)
    ours = apd(pred, joint_mask)
    for i, target in enumerate(mm_targets):
        if target is None or target.shape[0] < 2:
            out.append(nan)
            continue
        t = _apply_mask(target.to(pred.device), joint_mask)
        theirs = apd(t.unsqueeze(0), None)[0]
        out.append((ours[i] - theirs).abs())
    return torch.stack(out)


def multimodal_ade_fde(pred, mm_targets, joint_mask=None, conventional=False):
    mmade, mmfde = [], []
    nan = torch.tensor(float("nan"), device=pred.device, dtype=pred.dtype)
    for i, target in enumerate(mm_targets):
        if target is None or target.shape[0] == 0:

            mmade.append(nan)
            mmfde.append(nan)
            continue
        p = _apply_mask(pred[i], joint_mask)
        t = _apply_mask(target.to(p.device), joint_mask)
        diff = p[:, None] - t[None]
        if conventional:
            per_frame = diff.flatten(start_dim=-2).norm(dim=-1)
            scale = 1.0
        else:
            per_frame = diff.norm(dim=-1).mean(dim=-1)
            scale = CM

        mmade.append(per_frame.mean(dim=-1).min(dim=0).values.mean() * scale)
        mmfde.append(per_frame[..., -1].min(dim=0).values.mean() * scale)
    return torch.stack(mmade), torch.stack(mmfde)


def build_multimodal_gt(last_obs_poses, futures, threshold: float = 0.4):
    flat = last_obs_poses.reshape(last_obs_poses.shape[0], -1)
    dist = torch.cdist(flat, flat)
    neighbours = []
    for i in range(dist.shape[0]):
        idx = torch.nonzero(dist[i] < threshold, as_tuple=False).flatten()
        neighbours.append(idx[idx != i])
    return neighbours


def limb_metrics(pred, bone_lengths_gt, parents, eps: float = 1e-8):
    parent_pos = pred[..., parents.clamp(min=0), :]
    lengths = (pred - parent_pos).norm(dim=-1)[..., 1:]
    gt = bone_lengths_gt[..., 1:].clamp_min(eps)
    gt = gt[:, None, None]

    stretch = (gt - lengths).abs() / gt
    jitter = (lengths[..., 1:, :] - lengths[..., :-1, :]).abs() / gt

    return (
        stretch.mean(dim=(1, 2, 3)) * 100.0,
        jitter.mean(dim=(1, 2, 3)) * 100.0,
        stretch.pow(2).mean(dim=(1, 2, 3)).sqrt() * 100.0,
        jitter.pow(2).mean(dim=(1, 2, 3)).sqrt() * 100.0,
    )


def summarize_counts(values: dict) -> dict:
    out = {}
    for key, val in values.items():
        if torch.is_tensor(val):
            val = val.detach().float()
            val = val[~torch.isnan(val)]
            out[key] = (float(val.sum()), int(val.numel()))
        else:
            out[key] = (float(val), 1)
    return out


def summarize(values: dict) -> dict:
    out = {}
    for key, val in values.items():
        if torch.is_tensor(val):
            val = val.detach().float()
            val = val[~torch.isnan(val)]
            out[key] = float(val.mean()) if val.numel() else float("nan")
        else:
            out[key] = float(val)
    return out
