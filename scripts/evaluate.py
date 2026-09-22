from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import time

import numpy as np
import torch

from .data_processing.dataset import DataConfig, MotionWindowDataset
from .data_processing.skeleton import forward_kinematics
from .models.flow_matching import FlowMatchingConfig, GraphFlowMatching
from .models.gst_transformer import GSTConfig, GSTVelocityField
from .utils.dist import cleanup_distributed, setup_distributed
from .utils.published_metrics import (APD_THRESHOLD, MM_THRESHOLD, PublishedMetrics,
                                      class_label, neighbour_sets)
from .utils.metrics import (
    ade_fde, ade_fde_conventional, apd, apd_conventional, apde,
    build_multimodal_gt, limb_metrics, mae_joint_angle, mean_angle_error,
    multimodal_ade_fde, sd_limb_tables, summarize_counts,
)

FIXED_SETTINGS = {
    DataConfig: {"root_mode": "centered", "bone_length_source": "mean", "up_axis": 2,
                 "segments_csv": "", "augment_rotate": None},
    FlowMatchingConfig: {"sigma_root": None, "lambda_root": None, "lambda_dir": 1.0,
                         "predict_root": False, "time_sampler": "logit_normal",
                         "logit_normal_mean": 0.0, "logit_normal_std": 1.0,
                         "sigma_ramp_values": ()},
    GSTConfig: {"fixed_self_loops": True, "tie_spatial_temporal": False,
                "use_structural_features": True, "use_joint_embedding": False,
                "attn_dropout": 0.0, "predict_root": False},
}


def config_kwargs(cls, raw):
    known = {f.name for f in dataclasses.fields(cls)}
    fixed = FIXED_SETTINGS[cls]
    out = {}
    for key, value in raw.items():
        if key in known:
            out[key] = value
        elif key in fixed:
            if fixed[key] is not None and value != fixed[key]:
                raise ValueError(
                    f"{cls.__name__}.{key}={value!r} in the checkpoint; this release "
                    f"implements only {key}={fixed[key]!r}")
        else:
            raise ValueError(f"unknown {cls.__name__} field {key!r} in the checkpoint")
    return out


def load_model(checkpoint_path, skeleton, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    data_cfg = DataConfig(**config_kwargs(DataConfig, {**cfg["data"], "split": "test"}))

    from .models.gst_transformer import GST_PRESETS
    if cfg.get("resolved_model"):

        model_kwargs = dict(cfg["resolved_model"])
    else:
        model_kwargs = {**GST_PRESETS[cfg["model_preset"]], **cfg["model"],
                        "frame_period": 1.0 / data_cfg.fps}
        print("  (checkpoint predates resolved_model; rebuilding from preset "
              f"{cfg['model_preset']!r} -- verify it has not changed)")

        weights = ckpt.get("ema") or ckpt["model"]
        hop_rows = weights.get("field.hop_bias.weight")
        if hop_rows is not None and hop_rows.shape[0] - 1 != model_kwargs.get(
                "max_hop_bucket", GSTConfig.max_hop_bucket):
            model_kwargs["max_hop_bucket"] = int(hop_rows.shape[0]) - 1
            print(f"  (recovered max_hop_bucket={model_kwargs['max_hop_bucket']} "
                  f"from the checkpoint's hop_bias)")
    field = GSTVelocityField(GSTConfig(**config_kwargs(GSTConfig, model_kwargs)), skeleton)
    flow_cfg = FlowMatchingConfig(**config_kwargs(FlowMatchingConfig, cfg["flow"]))
    module = GraphFlowMatching(field, flow_cfg, skeleton)

    state = ckpt.get("ema") or ckpt["model"]
    module.load_state_dict(state)
    return module.to(device).eval(), cfg, ckpt.get("step", 0)



@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--solver", default="euler",
                        choices=["euler", "midpoint", "heun"])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--mm-threshold", type=float, default=0.4)
    parser.add_argument("--stride", type=int, default=0,
                        help="segment stride; 0 = non-overlapping observation windows")
    parser.add_argument("--max-segments", type=int, default=0)
    parser.add_argument("--start-offset", type=int, default=0,
                        help="first window start frame")
    parser.add_argument("--pool", default="stride", choices=["stride", "published"],
                        help="published: the BeLFusion/SkeletonDiffusion AMASS test "
                             "segmentation (--stride 120 --start-offset 150, every "
                             "window; pass --max-segments 0)")
    parser.add_argument("--chunk-size", type=int, default=256,
                        help="cap the expanded (batch x samples) per ODE solve; "
                             "raise it on a large card, 0 disables chunking")
    parser.add_argument("--data-path", default=None,
                        help="evaluate on a different dataset than the one the "
                             "checkpoint was trained on. The field's parameters "
                             "do not depend on the joint count and every graph "
                             "buffer is non-persistent, so a checkpoint loads "
                             "onto another kinematic tree unchanged -- this is "
                             "the zero-shot kinematics setting")
    parser.add_argument("--data-source-fps", type=int, default=0,
                        help="native frame rate of --data-path when it differs "
                             "from the protocol's; H36M is 50 Hz. Frames are "
                             "resampled so the temporal attention bias, which is "
                             "a function of the offset in seconds, stays valid")
    parser.add_argument("--data-subjects", nargs="*", default=None,
                        help="subject-keyed files only (H36M): the protocol test "
                             "split is S9 S11")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None)
    parser.add_argument("--num-workers", type=int, default=4,
                        help="dataloader workers; keep at or below the CPU "
                             "allocation minus one")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ctx = setup_distributed()
    device = ctx.device if ctx.enabled else args.device
    torch.manual_seed(args.seed)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    data_kwargs = dict(ckpt["config"]["data"])
    data_kwargs.update(split=args.split, augment_mirror=False, augment=False)
    if args.stride:
        data_kwargs["stride"] = args.stride
    else:
        data_kwargs["stride"] = data_kwargs.get("obs_length", 30)
    if args.max_segments:
        data_kwargs["max_segments"] = args.max_segments
    if args.pool == "published":
        data_kwargs["stride"], data_kwargs["start_offset"] = 120, 150
    elif args.start_offset:
        data_kwargs["start_offset"] = args.start_offset
    zero_shot = args.data_path is not None and args.data_path != data_kwargs["path"]
    trained_on = data_kwargs["path"]
    if args.data_path:
        data_kwargs["path"] = args.data_path
    if args.data_source_fps:
        data_kwargs["source_fps"] = args.data_source_fps
    if args.data_subjects:
        data_kwargs["subjects"] = tuple(args.data_subjects)
    dataset = MotionWindowDataset(DataConfig(**config_kwargs(DataConfig, data_kwargs)))
    module, cfg, train_step = load_model(args.checkpoint, dataset.skeleton, device)
    parents = dataset.parents.to(device)
    try:
        sd_tables = sd_limb_tables(dataset.skeleton.num_joints)
    except ValueError as exc:
        sd_tables = None
        if ctx.is_main:
            print(f"  (SkeletonDiffusion MAE unavailable: {exc})")
    if ctx.is_main:
        print(f"checkpoint    {args.checkpoint} (train step {train_step})")
        print(f"split         {args.split}: {len(dataset)} segments, "
              f"stride {data_kwargs['stride']}")
        if zero_shot:
            print(f"ZERO-SHOT KINEMATICS")
            print(f"              trained on {trained_on}")
            print(f"              evaluating on {data_kwargs['path']}, "
                  f"{dataset.skeleton.num_joints} joints"
                  + (f", resampled {args.data_source_fps} -> {data_kwargs['fps']} Hz"
                     if args.data_source_fps else "")
                  + (f", subjects {list(args.data_subjects)}"
                     if args.data_subjects else ""))
        print(f"sampling      {args.num_samples} futures, {args.num_steps} "
              f"{args.solver} steps")
        if ctx.enabled:
            print(f"world size    {ctx.world_size} ranks, "
                  f"~{len(dataset) // ctx.world_size} segments each")

    full_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.num_workers)
    all_last_obs, all_future_pos = [], []
    for batch in full_loader:
        all_last_obs.append(batch["past_pos"][:, -1])
        all_future_pos.append(batch["future_pos"])
    last_obs = torch.cat(all_last_obs)
    future_pos_all = torch.cat(all_future_pos)

    shard = list(range(ctx.rank, len(dataset), ctx.world_size))
    mm_index_full = build_multimodal_gt(last_obs, future_pos_all, args.mm_threshold)
    mm_index = [mm_index_full[i] for i in shard]
    subject_keyed = bool(data_kwargs.get("subjects"))
    pub_mm, pub_apd = neighbour_sets(
        last_obs, [MM_THRESHOLD["h36m" if subject_keyed else "amass"], APD_THRESHOLD])
    native_fps = data_kwargs.get("source_fps") or data_kwargs.get("fps", 60)
    ref_frames = int(round(future_pos_all.shape[1] * native_fps / data_kwargs.get("fps", 60)))
    published = PublishedMetrics(
        [class_label(dataset.sequence_name(i), subject_keyed) for i in range(len(dataset))],
        future_pos_all, pub_mm, pub_apd, ref_frames)
    if ctx.is_main:
        print(f"multimodal GT threshold {args.mm_threshold}: mean "
              f"{np.mean([len(i) for i in mm_index_full]):.1f} sequences per segment")
    loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, shard), batch_size=args.batch_size,
        shuffle=False, num_workers=max(args.num_workers // 2, 0))

    pool = sorted({dataset.sequence_name(i).split("/")[0] for i in range(len(dataset))})
    if ctx.is_main:
        print(f"test pool     {len(pool)} sub-datasets: {', '.join(pool)}")

    acc, seen = {}, 0
    per_dataset = {}
    horizon_len = int(data_kwargs.get("pred_length", 120))
    start = time.time()

    for bi, batch in enumerate(loader):
        past = batch["past"].to(device)
        lengths = batch["bone_lengths"].to(device)
        future_pos = batch["future_pos"].to(device)
        horizon = batch["future"].shape[1]

        samples = module.sample(past, lengths, horizon, num_steps=args.num_steps,
                                num_samples=args.num_samples, solver=args.solver,
                                chunk_size=args.chunk_size)
        b, n, t, j, _ = samples.shape
        pred_pos = forward_kinematics(samples.float(),
                                      lengths[:, None, None].expand(b, n, t, j),
                                      parents)

        published.update(pred_pos, [int(i) for i in batch["index"]])
        ade, fde = ade_fde(pred_pos, future_pos)
        c_ade, c_fde = ade_fde_conventional(pred_pos, future_pos)
        mae = mean_angle_error(pred_pos, future_pos, parents)

        mae_sd = (mae_joint_angle(pred_pos, future_pos, *sd_tables)
                  if sd_tables else torch.full_like(mae, float("nan")))
        div = apd(pred_pos)
        c_apd = apd_conventional(pred_pos)
        stretch, jit, stretch_rmse, jit_rmse = limb_metrics(pred_pos, lengths, parents)

        offset = bi * args.batch_size
        mm_targets = [
            future_pos_all[mm_index[offset + k]] if len(mm_index[offset + k]) else None
            for k in range(b)
        ]
        mmade, mmfde = multimodal_ade_fde(pred_pos, mm_targets)
        c_mmade, c_mmfde = multimodal_ade_fde(pred_pos, mm_targets,
                                              conventional=True)
        apde_v = apde(pred_pos, mm_targets)

        metrics = summarize_counts({"ade": ade, "fde": fde, "mae": mae,
                             "mae_sd": mae_sd,
                             "mmade": mmade, "mmfde": mmfde, "apd": div,
                             "apde": apde_v,
                             "conv_ade": c_ade, "conv_fde": c_fde,
                             "conv_apd": c_apd,
                             "conv_mmade": c_mmade, "conv_mmfde": c_mmfde,
                             "stretching": stretch, "jitter": jit,
                             "stretching_rmse": stretch_rmse,
                             "jitter_rmse": jit_rmse})

        for key, (batch_total, batch_n) in metrics.items():
            if not batch_n:
                continue
            total, count = acc.get(key, (0.0, 0))
            acc[key] = (total + batch_total, count + batch_n)
        seen += b

        for k in range(b):

            name = dataset.sequence_name(int(batch["index"][k])).split("/")[0]
            row = per_dataset.setdefault(name, {"n": 0, "ade": 0.0, "fde": 0.0,
                                                "apd": 0.0})
            row["n"] += 1
            row["ade"] += float(ade[k])
            row["fde"] += float(fde[k])
            row["apd"] += float(div[k])

        if bi % 10 == 0 and ctx.is_main:
            done = seen / max(len(shard), 1)
            mean = lambda k: acc[k][0] / max(acc[k][1], 1)
            print(f"  rank0 {seen}/{len(shard)} segments  "
                  f"ade {mean('ade'):.2f}  fde {mean('fde'):.2f}  "
                  f"apd {mean('apd'):.2f}  ({done*100:.0f}%)")

    if ctx.enabled:
        import torch.distributed as dist
        keys = sorted(acc)
        buf = torch.tensor([[acc[k][0], acc[k][1]] for k in keys],
                           dtype=torch.float64, device=device)
        dist.all_reduce(buf, op=dist.ReduceOp.SUM)
        acc = {k: (float(buf[i, 0]), int(buf[i, 1])) for i, k in enumerate(keys)}

        published.all_reduce(device)
        gathered = [None] * ctx.world_size
        dist.all_gather_object(gathered, per_dataset)
        merged = {}
        for part in gathered:
            for name, row in part.items():
                tgt = merged.setdefault(name, {"n": 0, "ade": 0.0, "fde": 0.0,
                                               "apd": 0.0})
                for k in tgt:
                    tgt[k] += row[k]
        per_dataset = merged

        total_seen = torch.tensor(float(seen), device=device)
        dist.all_reduce(total_seen, op=dist.ReduceOp.SUM)
        seen = int(total_seen.item())

    results = {k: (total / count if count else float("nan"))
               for k, (total, count) in acc.items()}
    for key in ("mmade", "mmfde", "apde", "conv_mmade", "conv_mmfde", "mae_sd"):
        results.setdefault(key, float("nan"))

    j_total = dataset.skeleton.num_joints
    moving = j_total / max(j_total - 1, 1)
    for key in ("ade", "fde", "mmade", "mmfde", "apd"):
        if key in results:
            results[f"{key}_moving"] = results[key] * moving
    results["moving_joint_factor"] = moving

    apd_cofactor = math.sqrt(j_total * horizon_len) / 100.0
    results["conv_apde"] = results["apde"] * apd_cofactor
    results["apd_cofactor"] = apd_cofactor
    results["mm_segments_with_gt"] = int(acc.get("mmade", (0.0, 0))[1])
    results["world_size"] = ctx.world_size
    results["published"] = published.results()
    results["segmentation"] = {"name": args.pool, "stride": data_kwargs["stride"],
                               "start_offset": data_kwargs.get("start_offset", 0)}
    results["num_segments"] = seen
    results["num_samples"] = args.num_samples
    results["num_steps"] = args.num_steps
    results["solver"] = args.solver
    results["sigma_dir"] = float(module.cfg.sigma_dir)
    results["seconds"] = time.time() - start
    results["checkpoint"] = os.path.abspath(args.checkpoint)
    results["train_step"] = train_step

    if not ctx.is_main:
        cleanup_distributed(ctx)
        return

    print("\n" + "=" * 74)
    print(f"{'ADE':>8}{'FDE':>9}{'MAE':>9}{'MMADE':>9}{'MMFDE':>9}"
          f"{'APD':>10}{'str':>10}{'jit':>10}")
    print(f"{results['ade']:8.2f}{results['fde']:9.2f}{results['mae']:9.3f}"
          f"{results['mmade']:9.2f}{results['mmfde']:9.2f}{results['apd']:10.3f}"
          f"{results['stretching']:10.4f}{results['jitter']:10.4f}")
    print("=" * 74)
    print("units: cm, cm, deg, cm, cm, m, %, %   "
          f"(MMGT available for {results['mm_segments_with_gt']}/{seen} segments)")

    print(f"MAE {results['mae_sd']:.3f} deg in SkeletonDiffusion's definition "
          f"(inter-limb angles)   vs {results['mae']:.3f} in ours (bone orientation)")
    print(f"unified over the {j_total - 1} MOVING joints, EquiFusion's convention "
          f"(x {moving:.4f}):  uADE {results['ade_moving']:.3f}  "
          f"uFDE {results['fde_moving']:.3f}  uAPD {results['apd_moving']:.3f}")
    print(f"APDE {results['apde']:.3f} cm   "
          f"(|our APD - multimodal-GT APD|; lower is better, and unlike APD it "
          f"punishes over-dispersion)")

    if results["mmade"] < results["ade"]:
        print(f"  NOTE  MMADE {results['mmade']:.2f} < ADE {results['ade']:.2f}. "
              f"Every published row has MMADE above ADE, but that holds only when "
              f"the multimodal pool is built from the whole split; on a subset it "
              f"can invert. If this is a full-protocol run, check the reduction.")

    print("\nconventional convention (comparable to published tables)")
    print(f"  {'ADE':>8}{'FDE':>9}{'MMADE':>9}{'MMFDE':>9}{'APDE':>9}{'APD':>10}")
    print(f"  {results['conv_ade']:8.3f}{results['conv_fde']:9.3f}"
          f"{results['conv_mmade']:9.3f}{results['conv_mmfde']:9.3f}"
          f"{results['conv_apde']:9.3f}{results['conv_apd']:10.3f}")
    pub = results["published"]
    print()
    print("SkeletonDiffusion definitions (the published MMADE/MMFDE/APDE/CMD columns)")
    print(f"  MMADE {pub['mmade']:.3f}  MMFDE {pub['mmfde']:.3f}  APD {pub['apd']:.3f}  "
          f"APDE {pub['apde']:.3f}  CMD {pub['cmd']:.3f}   ({pub['cmd_frames']} frames)")
    if args.pool == "published":
        print("  This is the published segmentation, so these columns sit next to the "
              "published")
        print("  tables directly; the ZeroVelocity row below is the check that they do.")
    else:
        print("  Report your own ZeroVelocity row next to these. The test split "
              "differs from the")
        print("  published one (segmentation, AMASS version, GRAB), so the baseline "
              "is the only")
        print("  anchor that makes the comparison honest.")

    print(f"\nper sub-dataset ({len(per_dataset)} in pool)")
    print(f"  {'dataset':<20}{'segments':>10}{'ADE':>9}{'FDE':>9}{'APD':>9}")
    for name in sorted(per_dataset):
        row = per_dataset[name]
        n = row["n"]
        print(f"  {name:<20}{n:>10}{row['ade']/n:>9.2f}{row['fde']/n:>9.2f}"
              f"{row['apd']/n:>9.3f}")
    results["per_dataset"] = {k: {m: (v[m] / v["n"] if m != "n" else v["n"])
                                  for m in v} for k, v in per_dataset.items()}
    results["pool"] = pool

    expected = {"DFaust", "DanceDB", "GRAB", "HUMAN4D", "SOMA", "SSM", "Transitions"}
    if args.split == "test" and dataset.skeleton.num_joints == 22:
        missing = sorted(expected - set(pool))
        if missing:
            print(f"\n  WARNING: the AMASS test protocol expects 7 sub-datasets; "
                  f"missing {', '.join(missing)}.")
            print("  MMADE/MMFDE and CMD are computed over the pool, so all of the")
            print("  numbers above differ from the published tables. Report the pool.")
            results["protocol_complete"] = False
        else:
            results["protocol_complete"] = True

    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint)))
    out = args.out or os.path.join(run_dir, f"eval_{args.split}.json")

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nmetrics -> {out}")

    cleanup_distributed(ctx)

if __name__ == "__main__":
    main()
