"""Offline steering error per session group (val from cache, test from PNG) for one checkpoint.

Usage: python tools/per_group_eval.py <checkpoint> <prepared_dataset> [device]
Prints MAE now / in turns per group, using the cached preprocessed images.
"""
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from racing.data import read_prepared
from racing.dataset import DrivingDataset
from racing.inference import load_checkpoint
from racing.cache import build_cache

checkpoint, dataset = sys.argv[1], sys.argv[2]
device = torch.device(sys.argv[3] if len(sys.argv) > 3 else "cpu")
model, cfg, _ = load_checkpoint(checkpoint, device)
_, splits = read_prepared(Path(dataset), cfg)
rows = splits["val"]+splits["test"]
cache = build_cache(splits["train"]+splits["val"], cfg, Path("data/cache"), 8)  # same key as training
pred, y = [], []
for part, part_cache in ((splits["val"], cache), (splits["test"], None)):
    loader = torch.utils.data.DataLoader(DrivingDataset(part, cfg, cache=part_cache), batch_size=256, num_workers=0)
    with torch.inference_mode():
        for x, speed, target, _ in loader:
            pred.append(model(x.to(device), speed.to(device)).float().cpu().numpy()[:, 0]); y.append(target.numpy()[:, 0])
pred, y = np.concatenate(pred), np.concatenate(y)
groups = defaultdict(list)
for i, r in enumerate(rows): groups[r["group_id"]].append(i)
print(f"{'group':14s} {'n':>6s} {'mae':>7s} {'turn_mae':>8s} {'bias':>7s}")
for g, idx in sorted(groups.items()):
    idx = np.array(idx); err = pred[idx]-y[idx]; turn = np.abs(y[idx]) > .2
    print(f"{g:14s} {len(idx):6d} {np.abs(err).mean():7.4f} {np.abs(err[turn]).mean() if turn.any() else float('nan'):8.4f} {err.mean():+7.4f}")
