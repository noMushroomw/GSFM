from __future__ import annotations

import argparse
import copy
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel

from .data_processing.dataset import DataConfig, build_dataloader
from .data_processing.skeleton import forward_kinematics
from .models.flow_matching import FlowMatchingConfig, GraphFlowMatching
from .models.gst_transformer import GST_PRESETS, GSTConfig, GSTVelocityField
from .utils.dist import all_reduce_mean, cleanup_distributed, setup_distributed
from .utils.logging_utils import RunLogger
from .utils.metrics import ade_fde, apd, limb_metrics, summarize



@dataclass
class TrainConfig:

    run_dir: str = "runs"
    run_name: str = "amass_base"
    seed: int = 0

    batch_size: int = 64
    max_steps: int = 94200
    lr: float = 2e-4
    min_lr_ratio: float = 0.02
    weight_decay: float = 0.01
    betas: tuple = (0.9, 0.95)
    warmup_steps: int = 2000
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    amp_dtype: str = "bfloat16"
    num_workers: int = 8

    log_every: int = 50
    eval_every: int = 2000
    save_every: int = 5000
    eval_batches: int = 20
    eval_num_samples: int = 10
    eval_num_steps: int = 20

    eval_chunk_size: int = 128

    model_preset: str = "base"
    model: dict = field(default_factory=dict)
    flow: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)

    data_valid: dict = field(default_factory=dict)
    resolved_model: dict = field(default_factory=dict)


def load_config(path, overrides):
    raw = {}
    if path:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    for item in overrides or []:
        key, _, value = item.partition("=")
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            pass
        node = raw
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return TrainConfig(**raw)


def cosine_lr(step, cfg: TrainConfig, total_steps):
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(cfg.warmup_steps, 1)
    progress = (step - cfg.warmup_steps) / max(total_steps - cfg.warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    scale = cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.lr * scale


class EMA:

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval().float()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module, step: int):

        decay = min(self.decay, (1 + step) / (10 + step))
        for ema_p, p in zip(self.shadow.parameters(), model.parameters()):
            ema_p.lerp_(p.detach().float(), 1.0 - decay)
        for ema_b, b in zip(self.shadow.buffers(), model.buffers()):
            ema_b.copy_(b)


def build(cfg: TrainConfig, device):
    data_cfg = DataConfig(**{**{"split": "train"}, **cfg.data})
    train_set, train_loader, train_sampler = build_dataloader(
        data_cfg, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers,
        distributed=int(os.environ.get("WORLD_SIZE", 1)) > 1, drop_last=True,
        seed=cfg.seed)

    val_cfg = DataConfig(**{**cfg.data, "split": "valid",
                            "augment_mirror": False, "augment": False,
                            **cfg.data_valid})

    val_set, val_loader, _ = build_dataloader(
        val_cfg, cfg.batch_size, shuffle=False,
        num_workers=min(2, cfg.num_workers),
        distributed=int(os.environ.get("WORLD_SIZE", 1)) > 1, drop_last=False)

    skeleton = train_set.skeleton
    model_kwargs = {**GST_PRESETS[cfg.model_preset], **cfg.model,
                    "frame_period": 1.0 / data_cfg.fps}
    gst_cfg = GSTConfig(**model_kwargs)
    field_net = GSTVelocityField(gst_cfg, skeleton).to(device)

    cfg.resolved_model = asdict(gst_cfg)

    flow_cfg = FlowMatchingConfig(**cfg.flow)
    module = GraphFlowMatching(field_net, flow_cfg, skeleton).to(device)
    return module, train_set, train_loader, train_sampler, val_set, val_loader, data_cfg



@torch.no_grad()
def validate(module, loader, data_cfg, device, cfg: TrainConfig, amp_dtype):
    module.eval()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    parents = loader.dataset.parents.to(device)
    acc = {}
    seen = 0
    for i, batch in enumerate(loader):
        if i >= cfg.eval_batches:
            break
        past = batch["past"].to(device, non_blocking=True)
        future = batch["future"].to(device, non_blocking=True)
        lengths = batch["bone_lengths"].to(device, non_blocking=True)
        future_pos = batch["future_pos"].to(device, non_blocking=True)

        with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
            loss, stats = module.loss(past, future, lengths)
            samples = module.sample(past, lengths, future.shape[1],
                                    num_steps=cfg.eval_num_steps,
                                    num_samples=cfg.eval_num_samples,
                                    chunk_size=cfg.eval_chunk_size)

        samples = samples.float()
        b, n, t, j, _ = samples.shape
        exp_lengths = lengths[:, None, None].expand(b, n, t, j)
        pred_pos = forward_kinematics(samples, exp_lengths, parents)

        ade, fde = ade_fde(pred_pos, future_pos)
        div = apd(pred_pos)
        stretch, jit, stretch_rmse, jit_rmse = limb_metrics(pred_pos, lengths, parents)

        metrics = summarize({
            "loss": stats["loss"].expand(1), "ade": ade, "fde": fde, "apd": div,
            "stretching": stretch, "jitter": jit,
            "stretching_rmse": stretch_rmse, "jitter_rmse": jit_rmse,
        })
        weight = past.shape[0]
        for key, value in metrics.items():
            acc[key] = acc.get(key, 0.0) + value * weight
        seen += weight

    module.train()
    return {k: v / max(seen, 1) for k, v in acc.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", nargs="*", dest="overrides",
                        help="dotted overrides, e.g. --set lr=1e-4 model.hidden=512")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--resume", default=None, help="path to a checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    if args.run_name:
        cfg.run_name = args.run_name

    ctx = setup_distributed()
    torch.manual_seed(cfg.seed + ctx.rank)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    device = ctx.device

    module, train_set, train_loader, train_sampler, val_set, val_loader, data_cfg = \
        build(cfg, device)

    amp_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                 "none": None}[cfg.amp_dtype]
    if device == "cpu":
        amp_dtype = None
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype is torch.float16)

    raw_module = module
    if ctx.enabled:
        module = DistributedDataParallel(module, device_ids=[ctx.local_rank]
                                         if device != "cpu" else None)

    ema = EMA(raw_module, cfg.ema_decay)

    NO_DECAY = ("iter_embed", "hop_bias", "rel_bias")

    def wants_decay(name, param):
        return param.ndim >= 2 and not any(k in name for k in NO_DECAY)

    decay_params = [p for n, p in raw_module.named_parameters()
                    if p.requires_grad and wants_decay(n, p)]
    nodecay_params = [p for n, p in raw_module.named_parameters()
                      if p.requires_grad and not wants_decay(n, p)]
    optimizer = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": cfg.weight_decay},
         {"params": nodecay_params, "weight_decay": 0.0}],
        lr=cfg.lr, betas=tuple(cfg.betas), eps=1e-8)

    total_steps = cfg.max_steps

    run_dir = os.path.join(cfg.run_dir, cfg.run_name)
    logger = RunLogger(run_dir, config=asdict(cfg), enabled=ctx.is_main,
                       run_name=cfg.run_name, resume=args.resume is not None)

    start_step = 0
    best_ade = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", 0)
        best_ade = ckpt.get("best_ade", float("inf"))
        if ckpt.get("ema") is not None:
            ema.shadow.load_state_dict(ckpt["ema"])
        if ctx.is_main:
            print(f"resumed from {args.resume} at step {start_step}")

    if ctx.is_main:
        params = raw_module.field.num_parameters()
        print(f"run           {run_dir}")
        print(f"skeleton      {train_set.skeleton.name}, "
              f"J={train_set.num_joints}, B={train_set.skeleton.num_bones}")
        print(f"segments      train {len(train_set)}, valid {len(val_set)}")
        print(f"model         {cfg.model_preset}, {params/1e6:.1f}M parameters")
        print(f"world size    {ctx.world_size}, batch/gpu {cfg.batch_size}, "
              f"effective {cfg.batch_size * ctx.world_size}")
        print(f"steps         {total_steps}")

    step = start_step
    module.train()
    t0 = time.time()
    running = {}

    epoch = 0
    while step < total_steps:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for batch in train_loader:
            past = batch["past"].to(device, non_blocking=True)
            future = batch["future"].to(device, non_blocking=True)
            lengths = batch["bone_lengths"].to(device, non_blocking=True)

            lr = cosine_lr(step, cfg, total_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            with torch.autocast("cuda", dtype=amp_dtype,
                                enabled=amp_dtype is not None):
                loss, stats = module(past, future, lengths)
            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(raw_module.parameters(),
                                                       cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            ema.update(raw_module, step)
            step += 1

            for key, value in stats.items():
                running[key] = running.get(key, 0.0) + float(value)
            running["grad_norm"] = running.get("grad_norm", 0.0) + float(grad_norm)

            if step % cfg.log_every == 0:
                scale = cfg.log_every
                payload = {k: v / scale for k, v in running.items()}
                payload["lr"] = lr
                payload["steps_per_sec"] = cfg.log_every / (time.time() - t0)
                running, t0 = {}, time.time()
                if ctx.is_main:
                    logger.log(step, split="train", epoch=epoch, **payload)
                    print(f"step {step:>7}/{total_steps}  loss {payload['loss']:.4f}  "
                          f"lr {lr:.2e}  {payload['steps_per_sec']:.2f} it/s")

            if step % cfg.eval_every == 0 or step == total_steps:
                target = ema.shadow
                metrics = validate(target, val_loader, data_cfg, device, cfg, amp_dtype)
                metrics = {k: all_reduce_mean(v, ctx) for k, v in metrics.items()}
                if ctx.is_main:
                    logger.log(step, split="valid", epoch=epoch, **metrics)
                    print("  valid  " + "  ".join(
                        f"{k} {v:.4f}" for k, v in sorted(metrics.items())))
                    if metrics["ade"] < best_ade:
                        best_ade = metrics["ade"]
                        torch.save({"model": raw_module.state_dict(),
                                    "ema": ema.shadow.state_dict(),
                                    "optimizer": optimizer.state_dict(),
                                    "step": step, "best_ade": best_ade,
                                    "config": asdict(cfg)},
                                   logger.checkpoint_path("best"))
                    logger.update_summary(best_ade=best_ade, last_step=step, **metrics)

            if ctx.is_main and (step % cfg.save_every == 0 or step == total_steps):
                torch.save({"model": raw_module.state_dict(),
                            "ema": ema.shadow.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "step": step, "best_ade": best_ade,
                            "config": asdict(cfg)},
                           logger.checkpoint_path("last"))

            if step >= total_steps:
                break

        epoch += 1

    if ctx.is_main:
        logger.close()
        print(f"done; best valid ADE {best_ade:.4f} cm")
    cleanup_distributed(ctx)

if __name__ == "__main__":
    main()
