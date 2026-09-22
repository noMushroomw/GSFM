from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import time
from dataclasses import asdict, is_dataclass


def _jsonable(obj):
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if hasattr(obj, "item") and getattr(obj, "ndim", 1) == 0:
        return obj.item()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
            text=True, timeout=5).strip()
    except Exception:
        return None


class RunLogger:

    def __init__(self, run_dir: str, config=None, enabled: bool = True,
                 run_name: str | None = None,
                 resume: bool = False):
        self.enabled = enabled
        self.run_dir = os.path.abspath(run_dir)
        self.start_time = time.time()
        self.summary = {}

        if not enabled:
            return

        os.makedirs(os.path.join(self.run_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(self.run_dir, "samples"), exist_ok=True)
        self.metrics_path = os.path.join(self.run_dir, "metrics.jsonl")
        self.summary_path = os.path.join(self.run_dir, "summary.json")
        self._fh = open(self.metrics_path, "a", buffering=1, encoding="utf-8")

        if os.path.exists(self.summary_path) and resume:
            with open(self.summary_path, encoding="utf-8") as f:
                self.summary = json.load(f)

        payload = {
            "config": _jsonable(config) if config is not None else {},
            "git_commit": _git_commit(),
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "created": time.time(),
            "run_name": run_name or os.path.basename(self.run_dir),
        }
        with open(os.path.join(self.run_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)


    def log(self, step: int, split: str = "train", epoch: int | None = None, **values):
        if not self.enabled:
            return
        record = {
            "step": int(step),
            "epoch": epoch,
            "split": split,
            "wall_time": time.time() - self.start_time,
            "timestamp": time.time(),
        }
        record.update({k: _jsonable(v) for k, v in values.items()})
        self._fh.write(json.dumps(record) + "\n")

    def update_summary(self, **values):
        if not self.enabled:
            return
        self.summary.update({k: _jsonable(v) for k, v in values.items()})
        tmp = self.summary_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.summary, f, indent=2)
        os.replace(tmp, self.summary_path)

    def checkpoint_path(self, tag: str) -> str:
        return os.path.join(self.run_dir, "checkpoints", f"ckpt_{tag}.pt")

    def sample_path(self, tag: str) -> str:
        return os.path.join(self.run_dir, "samples", f"{tag}.npz")

    def close(self):
        if not self.enabled:
            return
        self._fh.close()


def read_metrics(run_dir: str):
    path = os.path.join(run_dir, "metrics.jsonl")
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return records
