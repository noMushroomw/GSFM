from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

AMASS_JOINT_NAMES = [
    "pelvis",
    "left_hip", "right_hip", "spine1",
    "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3",
    "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
]
AMASS_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]

H36M17_JOINT_NAMES = [
    "Hip", "RHip", "RKnee", "RFoot", "LHip", "LKnee", "LFoot",
    "Spine", "Thorax", "Neck/Nose", "Head",
    "LShoulder", "LElbow", "LWrist", "RShoulder", "RElbow", "RWrist",
]

H36M17_PARENTS = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 14, 15]



@dataclass(frozen=True)
class Skeleton:

    name: str
    parents: tuple[int, ...]
    joint_names: tuple[str, ...]

    @classmethod
    def amass(cls) -> "Skeleton":
        return cls("amass", tuple(AMASS_PARENTS), tuple(AMASS_JOINT_NAMES))

    @classmethod
    def h36m17(cls) -> "Skeleton":
        return cls("h36m17", tuple(H36M17_PARENTS), tuple(H36M17_JOINT_NAMES))

    @classmethod
    def from_parents(cls, parents, joint_names=None, name="custom") -> "Skeleton":
        parents = tuple(int(p) for p in parents)
        if joint_names is None:
            joint_names = tuple(f"joint_{i}" for i in range(len(parents)))
        return cls(name, parents, tuple(joint_names))

    def __post_init__(self):
        p = self.parents
        if p[0] != -1:
            raise ValueError("joint 0 must be the root (parents[0] == -1)")
        for j in range(1, len(p)):
            if not (0 <= p[j] < j):
                raise ValueError(
                    f"parents must be topologically ordered; parents[{j}]={p[j]}"
                )

    @property
    def num_joints(self) -> int:
        return len(self.parents)

    @property
    def num_bones(self) -> int:
        return len(self.parents) - 1

    def parents_array(self) -> np.ndarray:
        return np.asarray(self.parents, dtype=np.int64)

    def edges(self) -> list[tuple[int, int]]:
        return [(self.parents[j], j) for j in range(1, self.num_joints)]

    def adjacency(self) -> np.ndarray:
        j = self.num_joints
        a = np.zeros((j, j), dtype=np.float32)
        for u, v in self.edges():
            a[u, v] = a[v, u] = 1.0
        return a

    def hop_distance(self) -> np.ndarray:
        j = self.num_joints
        big = j + 1
        d = np.full((j, j), big, dtype=np.int64)
        np.fill_diagonal(d, 0)
        for u, v in self.edges():
            d[u, v] = d[v, u] = 1
        for k in range(j):
            d = np.minimum(d, d[:, k, None] + d[None, k, :])
        return d

    def relation(self) -> np.ndarray:
        j = self.num_joints
        rel = np.full((j, j), 4, dtype=np.int64)
        for a in range(j):
            for b in range(j):
                if a == b:
                    rel[a, b] = 0
                elif self.parents[b] == a:
                    rel[a, b] = 1
                elif self.parents[a] == b:
                    rel[a, b] = 2
                elif a > 0 and b > 0 and self.parents[a] == self.parents[b]:
                    rel[a, b] = 3
        return rel

    def depth(self) -> np.ndarray:
        d = np.zeros(self.num_joints, dtype=np.int64)
        for j in range(1, self.num_joints):
            d[j] = d[self.parents[j]] + 1
        return d

    def num_children(self) -> np.ndarray:
        c = np.zeros(self.num_joints, dtype=np.int64)
        for j in range(1, self.num_joints):
            c[self.parents[j]] += 1
        return c

    def is_end_effector(self) -> np.ndarray:
        return (self.num_children() == 0).astype(np.int64)

    def structural_features(self) -> np.ndarray:
        return np.stack(
            [
                self.depth().astype(np.float32),
                self.num_children().astype(np.float32),
                self.is_end_effector().astype(np.float32),
                (self.parents_array() < 0).astype(np.float32),
            ],
            axis=-1,
        )


def forward_kinematics(state, bone_lengths, parents):
    j = state.shape[-2]

    tree = parents.tolist() if torch.is_tensor(parents) else list(parents)
    pos = [state[..., 0, :]]
    for joint in range(1, j):
        offset = state[..., joint, :] * bone_lengths[..., joint].unsqueeze(-1)
        pos.append(pos[tree[joint]] + offset)
    return torch.stack(pos, dim=-2)


def positions_to_manifold(positions, parents, eps: float = 1e-8):
    parent_pos = positions[..., parents.clamp(min=0), :]
    offsets = positions - parent_pos
    lengths = offsets.norm(dim=-1)
    dirs = offsets / lengths.clamp_min(eps).unsqueeze(-1)
    state = torch.cat([positions[..., :1, :], dirs[..., 1:, :]], dim=-2)
    lengths = torch.cat([torch.zeros_like(lengths[..., :1]), lengths[..., 1:]], dim=-1)
    return state, lengths


def numpy_forward_kinematics(state: np.ndarray, bone_lengths: np.ndarray,
                             parents: np.ndarray) -> np.ndarray:
    j = state.shape[-2]
    pos = np.empty_like(state)
    pos[..., 0, :] = state[..., 0, :]
    for joint in range(1, j):
        pos[..., joint, :] = (
            pos[..., parents[joint], :]
            + state[..., joint, :] * bone_lengths[..., joint, None]
        )
    return pos
