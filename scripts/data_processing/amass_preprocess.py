from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import torch

from .skeleton import AMASS_JOINT_NAMES, AMASS_PARENTS

NUM_BODY_JOINTS = 22
TARGET_FPS = 60

SPLITS = {
    "train": ["ACCAD", "BMLhandball", "BMLmovi", "BMLrub", "CMU", "EKUT",
              "EyesJapanDataset", "KIT", "PosePrior", "TCDHands", "TotalCapture"],
    "valid": ["HumanEva", "HDM05", "SFU", "MoSh"],
    "test":  ["DFaust", "DanceDB", "GRAB", "HUMAN4D", "SOMA", "SSM", "Transitions"],
}
ALIASES = {
    "PosePrior": ["MPI_Limits", "PosePrior", "MPILimits"],
    "HDM05": ["MPI_HDM05", "HDM05"],
    "MoSh": ["MPI_mosh", "MoSh", "MPImosh"],
    "SSM": ["SSM_synced", "SSM"],
    "Transitions": ["Transitions_mocap", "Transitions"],
    "EyesJapanDataset": ["Eyes_Japan_Dataset", "EyesJapanDataset"],
    "TCDHands": ["TCD_handMocap", "TCDHands"],
    "BMLrub": ["BioMotionLab_NTroje", "BMLrub"],
    "DFaust": ["DFaust_67", "DFaust", "DFaust67"],
    "HUMAN4D": ["HUMAN4D", "Human4D"],
}


def dataset_dir(root: str, name: str):
    candidates = [name] + ALIASES.get(name, [])
    lowered = {c.lower() for c in candidates}
    for entry in sorted(os.listdir(root)):
        if entry.lower() in lowered and os.path.isdir(os.path.join(root, entry)):
            path = os.path.join(root, entry)

            inner = [d for d in os.listdir(path)
                     if d.lower() in lowered and os.path.isdir(os.path.join(path, d))]
            return os.path.join(path, inner[0]) if inner else path
    return None


class SMPLHJoints:

    def __init__(self, model_path: str, device="cpu"):
        data = np.load(model_path, allow_pickle=True)
        self.device = device
        to = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)
        self.v_template = to(data["v_template"])
        shapedirs = np.asarray(data["shapedirs"])
        self.shapedirs = to(shapedirs[:, :, :16])
        self.J_regressor = to(data["J_regressor"])
        self.parents = torch.as_tensor(np.asarray(data["kintree_table"])[0].astype(np.int64),
                                       device=device)
        self.parents[0] = -1

    def rest_joints(self, betas):
        n_betas = self.shapedirs.shape[-1]
        betas = betas[:, :n_betas]
        verts = self.v_template[None] + torch.einsum("vck,bk->bvc", self.shapedirs, betas)
        return torch.einsum("jv,bvc->bjc", self.J_regressor, verts)

    @staticmethod
    def rodrigues(axis_angle):
        theta = axis_angle.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        axis = axis_angle / theta
        cos, sin = torch.cos(theta)[..., None], torch.sin(theta)[..., None]
        zero = torch.zeros_like(axis[:, :1])
        k = torch.cat([zero, -axis[:, 2:3], axis[:, 1:2],
                       axis[:, 2:3], zero, -axis[:, 0:1],
                       -axis[:, 1:2], axis[:, 0:1], zero], dim=1).view(-1, 3, 3)
        eye = torch.eye(3, device=axis_angle.device, dtype=axis_angle.dtype)[None]
        return eye + sin * k + (1 - cos) * (k @ k)

    def __call__(self, poses, betas, trans, num_joints=NUM_BODY_JOINTS):
        frames = poses.shape[0]
        rest = self.rest_joints(betas[None])[0, :num_joints]
        rots = self.rodrigues(poses[:, :num_joints * 3].reshape(-1, 3))
        rots = rots.view(frames, num_joints, 3, 3)

        global_rot = [rots[:, 0]]
        joints = [rest[0].expand(frames, 3)]
        for j in range(1, num_joints):
            parent = int(self.parents[j])
            offset = rest[j] - rest[parent]
            global_rot.append(global_rot[parent] @ rots[:, j])
            joints.append(joints[parent] + torch.einsum(
                "fab,b->fa", global_rot[parent], offset))
        return torch.stack(joints, dim=1) + trans[:, None]

SMPLH_LAYOUTS = [
    "smplh/{gender}/model.npz",
    "{gender}/model.npz",
    "smplh/SMPLH_{GENDER}.npz",
    "SMPLH_{GENDER}.npz",
    "smplh/smplh/{gender}/model.npz",
]


def parse_gender(value) -> str:
    if hasattr(value, "item") and getattr(value, "ndim", 0) == 0:
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8", "replace")
    text = str(value).strip().lower()
    for name in ("female", "neutral", "male"):
        if name in text:
            return name
    return "unknown"


def load_body_models(body_dir: str, device="cpu"):
    models = {}
    for gender in ("male", "female", "neutral"):
        for layout in SMPLH_LAYOUTS:
            path = os.path.join(body_dir, layout.format(gender=gender,
                                                        GENDER=gender.upper()))
            if os.path.exists(path):
                models[gender] = SMPLHJoints(path, device=device)
                print(f"  {gender:<8} {path}")
                break
    if not models:
        sys.exit(
            f"No SMPL+H model found under {body_dir}.  Tried:\n"
            + "\n".join(f"  {body_dir}/{l.format(gender='<gender>', GENDER='<GENDER>')}"
                        for l in SMPLH_LAYOUTS)
            + "\n\nDownload the 'Extended SMPL+H model' from "
              "https://mano.is.tue.mpg.de/ and extract it there.\n"
              "The archive is smplh.tar.xz; its female/ male/ neutral/ folders go "
              f"under {body_dir}/smplh/."
        )
    if "neutral" not in models:
        models["neutral"] = models.get("male") or next(iter(models.values()))
    return models


def convert_sequence(npz_path, models, device, min_frames, force_gender=None):
    raw = np.load(npz_path, allow_pickle=True)
    keys = set(raw.files)
    if not {"poses", "trans", "betas"} <= keys:
        return None, "not a mocap npz"

    fps_key = "mocap_framerate" if "mocap_framerate" in keys else "mocap_frame_rate"
    fps = float(raw[fps_key]) if fps_key in keys else 60.0
    stride = max(1, int(round(fps / TARGET_FPS)))
    if abs(fps / stride - TARGET_FPS) > 1.0:

        n_src = raw["poses"].shape[0]
        n_dst = int(round(n_src * TARGET_FPS / fps))
        if n_dst < min_frames:
            return None, f"too short ({n_dst} frames @60fps)"
        index = np.round(np.linspace(0, n_src - 1, n_dst)).astype(np.int64)
    else:
        index = np.arange(0, raw["poses"].shape[0], stride)
        if len(index) < min_frames:
            return None, f"too short ({len(index)} frames @60fps)"

    poses = torch.as_tensor(np.asarray(raw["poses"])[index], dtype=torch.float32,
                            device=device)
    trans = torch.as_tensor(np.asarray(raw["trans"])[index], dtype=torch.float32,
                            device=device)
    betas = torch.as_tensor(np.asarray(raw["betas"]), dtype=torch.float32, device=device)

    gender = parse_gender(raw["gender"]) if "gender" in keys else "neutral"

    if force_gender:
        gender = force_gender
    if gender not in models:
        return None, f"no body model for gender {gender!r}"
    model = models[gender]

    with torch.no_grad():
        joints = model(poses, betas, trans)
    return joints.cpu().numpy().astype(np.float32), gender


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/amass_raw/extracted")
    parser.add_argument("--body-models", default="data/body_models")
    parser.add_argument("--out", default="data/data_3d_amass.npz")
    parser.add_argument("--obs", type=int, default=30, help="observed frames @60fps")
    parser.add_argument("--pred", type=int, default=120, help="predicted frames @60fps")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit-per-dataset", type=int, default=0,
                        help="debug: keep at most N sequences per sub-dataset")
    parser.add_argument("--only", nargs="*", default=None,
                        help="process only these sub-datasets (use with --merge "
                             "to add a late arrival without redoing the rest)")
    parser.add_argument("--merge", action="store_true",
                        help="load --out first and update it instead of rebuilding")
    parser.add_argument("--neutral-datasets", nargs="*", default=None,
                        help="force the neutral body model for these sub-datasets. "
                             "Use it for any dataset taken from an 'SMPL+H N' "
                             "archive, whose betas are fitted to the neutral body "
                             "even when the gender field says otherwise.")
    args = parser.parse_args()

    raw_dir = os.path.abspath(args.raw_dir)
    if not os.path.isdir(raw_dir):
        sys.exit(f"{raw_dir} does not exist; run download_amass.py --extract first")

    models = load_body_models(os.path.abspath(args.body_models), device=args.device)
    print(f"loaded SMPL+H genders: {sorted(models)} on {args.device}")

    min_frames = args.obs + args.pred
    out = {split: {} for split in SPLITS}
    totals = {split: 0 for split in SPLITS}
    found = {split: [] for split in SPLITS}
    absent = {split: [] for split in SPLITS}
    skipped = 0

    if args.merge and os.path.exists(args.out):
        prior = np.load(args.out, allow_pickle=True)
        existing = prior["positions_3d"].item()
        for split in SPLITS:
            out[split].update(existing.get(split, {}))
            totals[split] = sum(v.shape[0] for v in out[split].values())
        print(f"merging into {args.out}: "
              + ", ".join(f"{s} {len(out[s])} seq" for s in SPLITS))

    force = {name: "neutral" for name in (args.neutral_datasets or [])}
    body_used = {}
    if args.merge and os.path.exists(args.out) and "body_models_used" in prior:
        body_used.update(prior["body_models_used"].item())
    if force:
        print(f"forcing the neutral body model for: {', '.join(sorted(force))}")

    only = set(args.only) if args.only else None
    for split, datasets in SPLITS.items():
        for name in datasets:
            if only is not None and name not in only:

                if any(k.startswith(name + "/") for k in out[split]):
                    found[split].append(name)
                else:
                    absent[split].append(name)
                continue

            for key in [k for k in out[split] if k.startswith(name + "/")]:
                totals[split] -= out[split].pop(key).shape[0]
            folder = dataset_dir(raw_dir, name)
            if folder is None:
                print(f"[{split}] {name}: NOT FOUND, skipping")
                absent[split].append(name)
                continue
            found[split].append(name)
            files = sorted(glob.glob(os.path.join(folder, "**", "*.npz"), recursive=True))
            files = [f for f in files if "shape.npz" not in os.path.basename(f)]
            if args.limit_per_dataset:
                files = files[: args.limit_per_dataset]
            kept = 0
            genders = {}
            for path in files:
                try:

                    joints, info = convert_sequence(path, models, args.device,
                                                    min_frames, force.get(name))
                except Exception as exc:
                    joints, info = None, f"{type(exc).__name__}: {exc}"
                if joints is None:
                    skipped += 1
                    continue
                genders[info] = genders.get(info, 0) + 1
                seq_id = f"{name}/{os.path.relpath(path, folder)}".replace("\\", "/")
                out[split][seq_id] = joints
                totals[split] += joints.shape[0]
                kept += 1
            body_used[name] = genders
            summary = ", ".join(f"{g} x{c}" for g, c in sorted(genders.items()))
            print(f"[{split}] {name}: {kept}/{len(files)} sequences, "
                  f"{sum(v.shape[0] for k, v in out[split].items() if k.startswith(name + '/'))}"
                  f" frames   [body: {summary or 'none'}]")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.savez_compressed(
        args.out,
        positions_3d=out,
        parents=np.asarray(AMASS_PARENTS, dtype=np.int64),
        joint_names=np.asarray(AMASS_JOINT_NAMES),
        fps=np.asarray(TARGET_FPS),
        splits=SPLITS,
        datasets_found=found,
        datasets_absent=absent,
        body_models_used=body_used,
    )
    print(f"\nwrote {args.out}")
    for split in SPLITS:
        print(f"  {split}: {len(found[split])}/{len(SPLITS[split])} datasets, "
              f"{len(out[split])} sequences, {totals[split]} frames")
    print(f"  skipped {skipped} files (too short or unreadable)")

    if any(absent.values()):
        print("\n" + "!" * 74)
        print("INCOMPLETE PROTOCOL -- these datasets are missing:")
        for split, names in absent.items():
            if names:
                print(f"  {split}: {', '.join(names)}")
        if absent["test"]:
            print("\n  A missing *test* dataset does more than drop its own rows: the")
            print("  multimodal ground truth (MMADE/MMFDE) and the CMD reference")
            print("  velocity are both built from the test split, so every test metric")
            print("  shifts.  Results from this file are not comparable to the")
            print("  published AMASS tables until the split is complete.")
        if absent["train"] or absent["valid"]:
            print("\n  Training and validation are also affected; retrain once complete.")
        print("!" * 74)

if __name__ == "__main__":
    main()
