"""Training is only performed by an explicit CLI invocation with --start-training."""
import argparse
import csv
import json
import random
import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from .config import load_config, contract_hash, dump_json, sha256
from .data import read_prepared, sampling_weights
from .dataset import DrivingDataset
from .model import RacingCNN


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def masked_loss(pred, target, mask, weights, beta):
    per = torch.nn.functional.smooth_l1_loss(pred, target, reduction="none", beta=beta)
    weighted = mask*weights
    return (per*weighted).sum()/weighted.sum().clamp_min(1e-8)


def atomic_checkpoint(path, payload):
    temp = path.with_suffix(".tmp")
    torch.save(payload, temp)
    os.replace(temp, path)


def train(cfg, dataset, out, cache_dir=None, gpu_data=False, max_epochs=None):
    t = cfg["training"]
    random.seed(t["seed"]); np.random.seed(t["seed"]); torch.manual_seed(t["seed"])
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(t["threads"])
    torch.use_deterministic_algorithms(t["deterministic"])
    torch.backends.cudnn.benchmark = not t["deterministic"]
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if t["device"] == "auto" else t["device"])
    dataset, out = Path(dataset), Path(out)
    manifest, all_rows = read_prepared(dataset, cfg)
    # Test labels are integrity-checked, but never used by training or model selection.
    rows = {name:all_rows[name] for name in ("train", "val")}
    # Optional training.subsample: keep every k-th sample (20 Hz frames are near duplicates);
    # this keeps the image cache small enough to live in GPU memory.
    k = t.get("subsample", 1)
    if k > 1: rows = {name:r[::k] for name,r in rows.items()}
    del all_rows
    out.mkdir(parents=True, exist_ok=False)
    dump_json(out/"config.json", cfg); dump_json(out/"dataset.json", manifest)
    dump_json(out/"environment.json", {"torch": str(torch.__version__), "numpy": np.__version__, "device": str(device)})
    cache = None
    if cache_dir:
        from .cache import build_cache
        cache = build_cache(rows["train"]+rows["val"], cfg, cache_dir, t["workers"])
    gen = torch.Generator().manual_seed(t["seed"])
    sampler = WeightedRandomSampler(sampling_weights(rows["train"], cfg), len(rows["train"]), replacement=True, generator=gen)
    if gpu_data:
        if cache is None or device.type != "cuda": raise ValueError("--gpu-data requires --cache-dir and a CUDA device")
        from .gpu_data import GPUBatches, load_cache_to_device
        images, index = load_cache_to_device(cache, device)
        loaders = {"train": GPUBatches(rows["train"], cfg, images, index, True, t["batch_size"], sampling_weights(rows["train"], cfg), t["seed"]),
                   "val": GPUBatches(rows["val"], cfg, images, index, False, t["batch_size"])}
    else:
        loaders = {name: DataLoader(DrivingDataset(r, cfg, name == "train", cache), batch_size=t["batch_size"], sampler=sampler if name == "train" else None, num_workers=t["workers"], worker_init_fn=seed_worker, generator=gen, pin_memory=device.type == "cuda") for name,r in rows.items()}
    model = RacingCNN(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=t["learning_rate"], weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, t["epochs"], eta_min=t["min_learning_rate"])
    amp = t["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    weights = torch.tensor(t["loss_weights"], device=device)
    best, stale = float("inf"), 0
    with (out/"metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "val_mae", "val_turn_mae", "lr"])
        writer.writeheader()
        for epoch in range(1, (max_epochs or t["epochs"])+1):
            stats = {}
            for name, loader in loaders.items():
                model.train(name == "train")
                total_loss = total_weight = abs_error = n = turn_error = turn_n = 0.0
                for x, speed, y, mask in loader:
                    x, speed, y, mask = (v.to(device) for v in (x, speed, y, mask))
                    with torch.set_grad_enabled(name == "train"):
                        with torch.autocast(device_type=device.type, enabled=amp):
                            pred = model(x, speed)
                            loss = masked_loss(pred, y, mask, weights, t["huber_beta"])
                        if not torch.isfinite(loss): raise RuntimeError("Non-finite loss; training aborted")
                        if name == "train":
                            optimizer.zero_grad(set_to_none=True)
                            scaler.scale(loss).backward()
                            scaler.unscale_(optimizer)
                            # With AMP, GradScaler skips steps whose gradients overflow and lowers its scale;
                            # without AMP a non-finite gradient is a real fault.
                            torch.nn.utils.clip_grad_norm_(model.parameters(), t["clip_grad_norm"], error_if_nonfinite=not amp)
                            scaler.step(optimizer); scaler.update()
                    denom = float((mask*weights).sum())
                    total_loss += float(loss.detach())*denom; total_weight += denom
                    err = (pred[:, 0]-y[:, 0]).abs().detach()
                    abs_error += float(err.sum()); n += len(x)
                    turns = y[:, 0].abs() > .2
                    turn_error += float(err[turns].sum()); turn_n += int(turns.sum())
                stats[name] = total_loss/max(total_weight, 1)
                if name == "val": mae, turn_mae = abs_error/n, turn_error/turn_n if turn_n else None
            writer.writerow({"epoch": epoch, "train_loss": stats["train"], "val_loss": stats["val"], "val_mae": mae, "val_turn_mae": turn_mae, "lr": optimizer.param_groups[0]["lr"]}); f.flush()
            scheduler.step()
            improved = stats["val"] < best-t["min_delta"]
            if improved: best, stale = stats["val"], 0
            else: stale += 1
            payload = {"format_version": 1, "trained": True, "completed_epochs": epoch, "config": cfg, "contract_hash": contract_hash(cfg), "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(), "validation_loss": stats["val"], "dataset_sha256": sha256(dataset/"dataset.json")}
            atomic_checkpoint(out/"last.pt", payload)
            if improved: atomic_checkpoint(out/"best.pt", payload)
            print(f"epoch={epoch} train={stats['train']:.5f} val={stats['val']:.5f} mae={mae:.5f}", flush=True)
            if stale >= t["patience"]: break
    dump_json(out/"checkpoint_hashes.json", {p.name: sha256(p) for p in (out/"best.pt", out/"last.pt")})


def main():
    p = argparse.ArgumentParser(description="Explicitly start CNN training (never automatic)")
    p.add_argument("--config", default="configs/race.json")
    p.add_argument("--dataset", default="data/prepared/v1")
    p.add_argument("--out", default="runs/cnn_v1")
    p.add_argument("--start-training", action="store_true")
    p.add_argument("--cache-dir", help="Optional folder for a one-time preprocessed image cache (faster epochs)")
    p.add_argument("--gpu-data", action="store_true", help="Keep the cache in GPU memory and augment batches on the GPU (needs --cache-dir)")
    p.add_argument("--max-epochs", type=int, help="Stop after this many epochs (speed tests); the LR schedule still uses training.epochs")
    a = p.parse_args()
    if not a.start_training: p.error("Training requires --start-training")
    train(load_config(a.config), a.dataset, a.out, a.cache_dir, a.gpu_data, a.max_epochs)


if __name__ == "__main__": main()
