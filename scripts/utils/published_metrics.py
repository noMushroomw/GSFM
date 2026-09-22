
from __future__ import annotations

import math

import torch


MM_THRESHOLD = {"amass": 0.4, "h36m": 0.5}
APD_THRESHOLD = 0.5


def class_label(sequence_name, subject_keyed):
    parts = sequence_name.split("/")
    return parts[1].split(" ")[0] if subject_keyed else parts[0]


def resample_time(x, n_out, dim):
    n_in = x.shape[dim]
    if n_out == n_in:
        return x
    src = torch.linspace(0.0, n_in - 1, n_out, dtype=torch.float64)
    lo = src.floor().long()
    hi = (lo + 1).clamp(max=n_in - 1)
    shape = [1] * x.dim()
    shape[dim] = n_out
    w = (src - lo.double()).to(device=x.device, dtype=x.dtype).view(shape)
    return (x.index_select(dim, lo.to(x.device)) * (1 - w)
            + x.index_select(dim, hi.to(x.device)) * w)


def neighbour_sets(last_obs, thresholds, block=1024):
    flat = last_obs.reshape(last_obs.shape[0], -1).float()
    out = [[] for _ in thresholds]
    for s in range(0, flat.shape[0], block):
        d = torch.cdist(flat[s:s + block], flat)
        for k, thr in enumerate(thresholds):
            out[k].extend(torch.nonzero(row < thr).flatten() for row in d)
    return out


def _mm_errors(pred_flat, gt_flat, chunk=64):
    ade, fde = [], []
    for s in range(0, gt_flat.shape[0], chunk):
        dist = (pred_flat[None] - gt_flat[s:s + chunk, None]).norm(dim=-1)
        ade.append(dist.mean(dim=-1).min(dim=1).values)
        fde.append(dist[..., -1].min(dim=1).values)
    return torch.cat(ade), torch.cat(fde)


def _apd_flat(x):
    m = x.shape[0]
    if m < 2:
        return 0.0
    flat = x.reshape(m, -1).double()
    d = torch.cdist(flat, flat)
    return float(d.sum() / (m * (m - 1)))


class PublishedMetrics:

    def __init__(self, classes, future_all, mm_incl, apd_incl, cmd_frames):
        self.names = sorted(set(classes))
        index = {c: i for i, c in enumerate(self.names)}
        self.seg_class = [index[c] for c in classes]
        self.future_all = future_all
        self.mm_incl = mm_incl
        self.apd_incl = apd_incl
        self.cmd_frames = cmd_frames

        fut = resample_time(future_all.double(), cmd_frames, dim=1)
        seg_motion = (fut[:, 1:, 1:] - fut[:, :-1, 1:]).norm(dim=-1).mean(dim=(1, 2))
        cls = torch.tensor(self.seg_class)
        n = torch.zeros(len(self.names), dtype=torch.float64).index_add_(0, cls, torch.ones(len(cls), dtype=torch.float64))
        self.ref = torch.zeros(len(self.names), dtype=torch.float64).index_add_(0, cls, seg_motion) / n.clamp_min(1)

        c = len(self.names)
        self.motion_sum = torch.zeros(c, cmd_frames - 1, dtype=torch.float64)
        self.class_n = torch.zeros(c, dtype=torch.float64)
        self.sums = torch.zeros(7, dtype=torch.float64)

    @torch.no_grad()
    def update(self, pred_pos, segment_ids):
        device = pred_pos.device
        b, n, t = pred_pos.shape[:3]
        flat = pred_pos.float().reshape(b, n, t, -1)
        pred = pred_pos.double()
        pr = resample_time(pred, self.cmd_frames, dim=2)
        for k, seg in enumerate(segment_ids):
            gts = self.future_all[self.mm_incl[seg]].to(device).float()
            e_ade, e_fde = _mm_errors(flat[k], gts.reshape(gts.shape[0], t, -1))
            self.sums[0] += float(e_ade.double().mean())
            self.sums[1] += float(e_fde.double().mean())
            self.sums[2] += 1
            pred_apd = _apd_flat(pr[k])
            self.sums[5] += pred_apd
            self.sums[6] += 1
            ref = resample_time(self.future_all[self.apd_incl[seg]].to(device).double(),
                                self.cmd_frames, dim=1)
            gt_apd = _apd_flat(ref)
            if gt_apd > 0:
                self.sums[3] += abs(pred_apd - gt_apd)
                self.sums[4] += 1

        motion = (pr[:, :, 1:, 1:] - pr[:, :, :-1, 1:]).norm(dim=-1).mean(dim=(1, 3))
        cls = torch.tensor([self.seg_class[s] for s in segment_ids])
        self.motion_sum.index_add_(0, cls, motion.cpu())
        self.class_n.index_add_(0, cls, torch.ones(len(cls), dtype=torch.float64))

    def all_reduce(self, device):
        import torch.distributed as dist
        for name in ("motion_sum", "class_n", "sums"):
            buf = getattr(self, name).to(device)
            dist.all_reduce(buf, op=dist.ReduceOp.SUM)
            setattr(self, name, buf.cpu())

    def results(self):
        s = self.sums
        total = float(self.class_n.sum())
        weights = torch.arange(self.cmd_frames - 1, 0, -1, dtype=torch.float64)
        cmd = 0.0
        for c in range(len(self.names)):
            if self.class_n[c] == 0:
                continue
            m_t = self.motion_sum[c] / self.class_n[c]
            cmd += float((weights * (m_t - self.ref[c]).abs()).sum()) * float(self.class_n[c]) / total
        return {"mmade": float(s[0] / s[2]) if s[2] else math.nan,
                "mmfde": float(s[1] / s[2]) if s[2] else math.nan,
                "apde": float(s[3] / s[4]) if s[4] else math.nan,
                "apd": float(s[5] / s[6]) if s[6] else math.nan,
                "cmd": cmd,
                "apde_segments": int(s[4]),
                "cmd_frames": self.cmd_frames,
                "class_segments": {c: int(self.class_n[i]) for i, c in enumerate(self.names)},
                "class_reference_motion_m": {c: float(self.ref[i]) for i, c in enumerate(self.names)},
                "definition": "SkeletonDiffusion src/metrics (multimodal set includes the "
                              "segment; APDE skips singleton sets; per-class CMD in metres)"}
