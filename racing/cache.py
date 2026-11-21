"""One-time cache of cropped/resized uint8 images, pixel-identical to prepare_image.

Training otherwise decodes every 255x255 PNG and resizes it at every epoch, which
keeps the GPU waiting. The cache is keyed by the preprocessing settings and the
image paths + SHA256, so a changed image or crop never reuses stale pixels.
"""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .preprocess import prepare_image


class _Decode(torch.utils.data.Dataset):
    def __init__(self, paths, cfg): self.paths, self.cfg = paths, cfg
    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            return torch.from_numpy(np.array(prepare_image(im.convert("RGB"), self.cfg), dtype=np.uint8))


def build_cache(rows, cfg, directory, workers=0):
    """Return (npy_path, {image_path: index}); builds the memmap only when missing."""
    digests = {}
    for r in rows:
        for path, digest in zip(r["images"], r["image_sha256"]):
            if digests.setdefault(path, digest) != digest: raise ValueError(f"Conflicting hashes for {path}")
    paths = sorted(digests)
    p = cfg["preprocess"]
    key = hashlib.sha256(json.dumps({"preprocess": p, "images": [[x, digests[x]] for x in paths]}, sort_keys=True).encode()).hexdigest()[:20]
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    target = directory/f"images_{key}.npy"
    if not target.exists():
        temp = directory/f"images_{key}.tmp.npy"
        array = np.lib.format.open_memmap(temp, mode="w+", dtype=np.uint8, shape=(len(paths), p["height"], p["width"], 3))
        loader = torch.utils.data.DataLoader(_Decode(paths, cfg), batch_size=256, num_workers=workers)
        start = 0
        for batch in loader:
            array[start:start+len(batch)] = batch.numpy(); start += len(batch)
            print(f"cache {start}/{len(paths)}", flush=True)
        if start != len(paths): raise RuntimeError("Incomplete image cache")
        array.flush(); del array
        os.replace(temp, target)
    return str(target), {x: i for i, x in enumerate(paths)}
