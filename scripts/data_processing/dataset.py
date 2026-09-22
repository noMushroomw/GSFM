from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .skeleton import Skeleton, positions_to_manifold



@dataclass
class DataConfig:
    path: str = "data/data_3d_amass.npz"
    split: str = "train"
    obs_length: int = 30
    pred_length: int = 120
    stride: int = 30
    fps: int = 60

    augment_mirror: bool = False
    augment: bool = True
    max_segments: int = 0
    source_fps: int = 0
    subjects: tuple = ()
    start_offset: int = 0

AMASS_MIRROR_PAIRS = [(1, 2), (4, 5), (7, 8), (10, 11), (13, 14),
                      (16, 17), (18, 19), (20, 21)]


class MotionWindowDataset(Dataset):
    def __init__(self, cfg: DataConfig, skeleton: Skeleton | None = None):
        self.cfg = cfg
        blob = np.load(cfg.path, allow_pickle=True)
        self.parents = torch.from_numpy(np.asarray(blob["parents"], dtype=np.int64))
        joint_names = [str(n) for n in np.asarray(blob["joint_names"])]
        self.skeleton = skeleton or Skeleton.from_parents(
            self.parents.tolist(), joint_names, name="dataset")

        positions = blob["positions_3d"].item()
        if cfg.split in positions:
            sequences = positions[cfg.split]
        else:
            keep = set(cfg.subjects) if cfg.subjects else None
            if keep is not None:
                missing = keep - set(positions)
                if missing:
                    raise ValueError(
                        f"subjects {sorted(missing)} are not in {cfg.path}; it has "
                        f"{sorted(positions)}")
            sequences = {f"{s}/{a}": arr
                         for s, actions in positions.items()
                         if keep is None or s in keep
                         for a, arr in actions.items()}

        self.sequences = []
        self.index = []
        window = cfg.obs_length + cfg.pred_length
        for name, arr in sorted(sequences.items()):
            arr = np.asarray(arr, dtype=np.float32)
            arr = self._resample(arr, cfg)
            if arr.shape[0] < window:
                continue
            seq_id = len(self.sequences)
            self.sequences.append((name, arr))
            for start in range(cfg.start_offset, arr.shape[0] - window + 1, cfg.stride):
                self.index.append((seq_id, start))

        if cfg.max_segments and cfg.max_segments < len(self.index):

            step = len(self.index) / cfg.max_segments
            self.index = [self.index[int(i * step)] for i in range(cfg.max_segments)]

        self.mirror_perm = self._mirror_permutation()

    @staticmethod
    def _resample(arr, cfg):
        if not cfg.source_fps or cfg.source_fps == cfg.fps:
            return arr
        n_in = arr.shape[0]
        n_out = int(round(n_in * cfg.fps / cfg.source_fps))
        if n_out < 2 or n_in < 2:
            return arr
        src = np.linspace(0.0, n_in - 1, n_out)
        lo = np.floor(src).astype(np.int64)
        hi = np.minimum(lo + 1, n_in - 1)
        w = (src - lo).astype(np.float32)[:, None, None]
        return (1.0 - w) * arr[lo] + w * arr[hi]

    def _mirror_permutation(self):
        perm = list(range(self.skeleton.num_joints))
        if self.skeleton.num_joints == 22:
            for a, b in AMASS_MIRROR_PAIRS:
                perm[a], perm[b] = perm[b], perm[a]
        return np.asarray(perm, dtype=np.int64)

    def __len__(self):
        return len(self.index)

    @property
    def num_joints(self):
        return self.skeleton.num_joints

    def sequence_name(self, i):
        return self.sequences[self.index[i][0]][0]

    def _augment(self, window, rng):
        if self.cfg.augment_mirror and rng.random() < 0.5:
            window = window[:, self.mirror_perm].copy()
            window[..., 0] *= -1.0
        angle = rng.uniform(0.0, 2.0 * np.pi)
        cos, sin = np.cos(angle), np.sin(angle)
        u, v = window[..., 0].copy(), window[..., 1].copy()
        window[..., 0] = cos * u - sin * v
        window[..., 1] = sin * u + cos * v
        return window

    def __getitem__(self, i):
        cfg = self.cfg
        seq_id, start = self.index[i]
        _, arr = self.sequences[seq_id]
        window = arr[start: start + cfg.obs_length + cfg.pred_length].copy()

        if cfg.split == "train" and cfg.augment:
            window = self._augment(window, np.random.default_rng())

        window = window - window[:, :1]

        positions = torch.from_numpy(window)
        state, lengths = positions_to_manifold(positions, self.parents)

        obs = cfg.obs_length
        bone_lengths = lengths[:obs].mean(0)

        return {
            "past": state[:obs],
            "future": state[obs:],
            "bone_lengths": bone_lengths,
            "past_pos": positions[:obs],
            "future_pos": positions[obs:],
            "index": torch.tensor(i, dtype=torch.long),
        }


def build_dataloader(cfg: DataConfig, batch_size: int, shuffle: bool,
                     num_workers: int = 4, distributed: bool = False,
                     drop_last: bool | None = None, seed: int = 0):
    from torch.utils.data import DataLoader, DistributedSampler

    dataset = MotionWindowDataset(cfg)
    sampler = None
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle, seed=seed,
                                     drop_last=bool(drop_last))
        shuffle = False
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=bool(drop_last) if drop_last is not None else shuffle,
        persistent_workers=num_workers > 0,
    )
    return dataset, loader, sampler
