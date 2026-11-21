"""Training batches built entirely on the GPU from the preprocessed image cache.

The whole cache (uint8, already cropped/resized by prepare_image) is copied once
into GPU memory. Each batch is gathered, augmented and normalized there, so the
CPU only shuffles indices. Validation (augment=False) gives exactly the same
tensors as DrivingDataset. Training augmentation follows PIL's Brightness then
Contrast formulas, flip and Gaussian noise, without PIL's intermediate uint8
rounding: the augmentation is random anyway, inference is unchanged.
"""
import numpy as np
import torch

_LUMA = (0.299, 0.587, 0.114)  # PIL "L" conversion used by ImageEnhance.Contrast


def load_cache_to_device(cache, device, chunk=4096):
    path, index = cache
    array = np.load(path, mmap_mode="r")
    images = torch.empty((array.shape[0], 3, array.shape[1], array.shape[2]), dtype=torch.uint8, device=device)
    for start in range(0, array.shape[0], chunk):
        block = torch.from_numpy(np.array(array[start:start+chunk]))
        images[start:start+len(block)] = block.to(device).permute(0, 3, 1, 2)
    return images, index


class GPUBatches:
    def __init__(self, rows, cfg, images, index, augment, batch_size, weights=None, seed=0):
        device = images.device
        self.cfg, self.images, self.augment, self.batch_size = cfg, images, augment, batch_size
        self.frames = torch.tensor([[index[p] for p in r["images"]] for r in rows], dtype=torch.long, device=device)
        pedal = bool(cfg["model"].get("throttle_output"))
        self.steering = torch.tensor([r["steering"]+([r["pedal"]] if pedal else []) for r in rows], dtype=torch.float32, device=device)
        self.mask = torch.tensor([r["mask"]+([1.0] if pedal else []) for r in rows], dtype=torch.float32, device=device)
        self.flip_sign = torch.tensor([-1.0]*len(cfg["model"]["horizons_s"])+([1.0] if pedal else []), device=device)
        self.speed = torch.tensor([[r["speed"]/cfg["model"]["speed_scale"]] for r in rows], dtype=torch.float32, device=device)
        self.weights = None if weights is None else torch.as_tensor(np.asarray(weights), dtype=torch.double)
        self.order_gen = torch.Generator().manual_seed(seed)
        self.aug_gen = torch.Generator(device=device).manual_seed(seed+1)
        self.luma = torch.tensor(_LUMA, device=device).view(1, 1, 3, 1, 1)

    def __len__(self): return (len(self.frames)+self.batch_size-1)//self.batch_size

    def _order(self):
        n = len(self.frames)
        if self.weights is not None:  # same semantics as WeightedRandomSampler(replacement=True)
            return torch.multinomial(self.weights, n, replacement=True, generator=self.order_gen)
        return torch.arange(n)

    def _augment(self, img, y):
        d, g, b = self.cfg["data"], self.aug_gen, img.shape[0]
        def uniform(spread): return 1-spread+2*spread*torch.rand((b, 1, 1, 1, 1), generator=g, device=img.device)
        img = (img*uniform(d["brightness"])).clamp(0, 255)
        mean = (img*self.luma).sum(2, keepdim=True).mean((3, 4), keepdim=True).round()
        img = (mean+uniform(d["contrast"])*(img-mean)).clamp(0, 255)
        flip = torch.rand(b, generator=g, device=img.device) < d["flip_probability"]
        img = torch.where(flip.view(b, 1, 1, 1, 1), img.flip(-1), img)
        y = torch.where(flip.view(b, 1), y*self.flip_sign, y)  # a mirrored image mirrors the steering, not the pedal
        return img, y

    def __iter__(self):
        for batch in self._order().split(self.batch_size):
            batch = batch.to(self.frames.device)
            img = self.images[self.frames[batch]].float()  # B, frames, 3, H, W in 0..255
            y = self.steering[batch]
            if self.augment: img, y = self._augment(img, y)
            x = (img/127.5-1).flatten(1, 2)  # same channel order as stack_images
            if self.augment and self.cfg["data"]["noise_std"]:
                noise = torch.randn(x.shape, generator=self.aug_gen, device=x.device)
                x = (x+noise*self.cfg["data"]["noise_std"]).clamp(-1, 1)
            yield x, self.speed[batch], y, self.mask[batch]
