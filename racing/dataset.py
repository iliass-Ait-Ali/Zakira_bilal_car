import numpy as np
import torch
from PIL import Image
from .preprocess import stack_images


class DrivingDataset(torch.utils.data.Dataset):
    def __init__(self, rows, cfg, augment=False, cache=None):
        """cache: optional (npy_path, {image_path: index}) from racing.cache.build_cache."""
        self.rows, self.cfg, self.augment = rows, cfg, augment
        self.cache_path, self.cache_index = cache if cache else (None, None)
        self._array = None  # opened lazily, once per DataLoader worker

    def __len__(self): return len(self.rows)

    def _images(self, row):
        if self.cache_path is None:
            images = []
            for path in row["images"]:
                with Image.open(path) as im: images.append(im.convert("RGB"))
            return images
        if self._array is None: self._array = np.load(self.cache_path, mmap_mode="r")
        return [Image.fromarray(np.array(self._array[self.cache_index[path]])) for path in row["images"]]

    def __getitem__(self, i):
        row = self.rows[i]
        images = self._images(row)
        rng = np.random.default_rng(int(np.random.randint(0, 2**31))) if self.augment else None
        x, flipped = stack_images(images, self.cfg, rng, prepared=self.cache_path is not None)
        y = np.asarray(row["steering"], dtype=np.float32)*( -1 if flipped else 1)
        mask = list(row["mask"])
        if self.cfg["model"].get("throttle_output"):  # the pedal is not mirrored
            y, mask = np.append(y, np.float32(row["pedal"])), mask+[1.0]
        speed = np.array([row["speed"]/self.cfg["model"]["speed_scale"]], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(speed), torch.from_numpy(y), torch.tensor(mask, dtype=torch.float32)
