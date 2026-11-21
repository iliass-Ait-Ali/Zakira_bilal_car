"""Lightweight, offline environment and trained-checkpoint verification."""
import argparse
import importlib.metadata
import json
import platform
import torch
from .config import load_config, sha256
from .model import RacingCNN


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/race.json")
    p.add_argument("--checkpoint")
    p.add_argument("--dataset", help="Check all split provenance/labels and forward real images without training")
    p.add_argument("--out", help="Optional JSON verification report")
    a = p.parse_args()
    cfg = load_config(a.config)
    payload = None
    if a.checkpoint:
        from .inference import load_checkpoint
        model, cfg, payload = load_checkpoint(a.checkpoint)
    else: model = RacingCNN(cfg).eval()
    torch.set_num_threads(2)
    x = torch.zeros(1, 3*cfg["model"]["frames"], cfg["preprocess"]["height"], cfg["preprocess"]["width"])
    with torch.inference_mode(): result = model(x, torch.zeros(1,1))
    if not torch.isfinite(result).all(): raise ValueError("Invalid model output")
    report = {"python": platform.python_version(), "packages": {n: importlib.metadata.version(n) for n in ("torch", "numpy", "Pillow", "opencv-python", "matplotlib")}, "cuda_available": torch.cuda.is_available(), "parameters": sum(p.numel() for p in model.parameters()), "output_shape": list(result.shape), "check": "synthetic forward only; not a driving evaluation"}
    if payload: report.update(checkpoint_sha256=sha256(a.checkpoint), completed_epochs=payload["completed_epochs"], validation_loss=payload["validation_loss"])
    if a.dataset:
        from .data import read_prepared
        from .dataset import DrivingDataset
        import numpy as np
        manifest, splits = read_prepared(a.dataset, cfg)
        checks={}
        for name,rows in splits.items():
            # Stratified locations across the split; pure forward, no gradients/optimizer.
            indices=np.linspace(0,len(rows)-1,min(64,len(rows)),dtype=int)
            ds=DrivingDataset([rows[int(i)] for i in indices],cfg,False)
            x,speed,target,mask=next(iter(torch.utils.data.DataLoader(ds,batch_size=64)))
            with torch.inference_mode(): prediction=model(x,speed)
            if prediction.shape != target.shape or not all(torch.isfinite(v).all() for v in (x,speed,target,mask,prediction)):
                raise ValueError("Real dataset/model compatibility check failed")
            checks[name]={"samples":len(rows), "groups":sorted({r["group_id"] for r in rows}),
                          "sources":sorted({r["source"] for r in rows}), "input_shape":list(x.shape),
                          "speed_shape":list(speed.shape), "target_shape":list(target.shape), "output_shape":list(prediction.shape),
                          "horizon_supervision":np.array([r["mask"] for r in rows]).sum(axis=0).astype(int).tolist()}
        report.update(dataset_checks=checks, check="all split metadata checked; real-image forward only; no training or driving evaluation")
    if a.out:
        from .config import dump_json
        dump_json(a.out,report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
