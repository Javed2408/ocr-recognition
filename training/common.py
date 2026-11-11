"""Shared helpers for the training scripts (run them from the project root: python -m training.<script>)."""
import json
import os
import random
import sys
import time

import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from cheque_forensics.config import METRICS_DIR  # noqa: E402


def seed_everything(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_metrics(name, metrics):
    os.makedirs(METRICS_DIR, exist_ok=True)
    path = os.path.join(METRICS_DIR, f"{name}.json")
    metrics = dict(metrics, generated_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"metrics written to {path}")
    return path


def parallel_map(fn, items, workers=None, chunksize=4):
    """multiprocessing map that also works on Windows (spawn)."""
    from multiprocessing import get_context
    workers = workers or max(1, min(16, (os.cpu_count() or 2) - 2))
    with get_context("spawn").Pool(workers) as pool:
        return pool.map(fn, items, chunksize=chunksize)


class Timer:
    def __init__(self):
        self.t = time.time()

    def __call__(self):
        return f"{time.time() - self.t:.0f}s"
