"""Exactly the same RGB crop/resize/normalization at training and inference."""
import numpy as np
from PIL import Image, ImageEnhance


def prepare_image(image, cfg):
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected uint8 RGB HWC image")
        image = Image.fromarray(image)
    image = image.convert("RGB")
    w, h = image.size
    if min(w, h) < 16:
        raise ValueError("Invalid source dimensions")
    p = cfg["preprocess"]
    image = image.crop((0, int(h*p["crop_top"]), w, int(h*(1-p["crop_bottom"]))))
    return image.resize((p["width"], p["height"]), Image.Resampling.BILINEAR)


def stack_images(images, cfg, rng=None, prepared=False):
    """prepared=True means the images already went through prepare_image (training cache)."""
    if len(images) != cfg["model"]["frames"]:
        raise ValueError("Wrong history length")
    images = images if prepared else [prepare_image(im, cfg) for im in images]
    flip = False
    if rng is not None:
        d = cfg["data"]
        flip = rng.random() < d["flip_probability"]
        brightness = rng.uniform(1-d["brightness"], 1+d["brightness"])
        contrast = rng.uniform(1-d["contrast"], 1+d["contrast"])
        images = [ImageEnhance.Contrast(ImageEnhance.Brightness(im).enhance(brightness)).enhance(contrast) for im in images]
        if flip: images = [im.transpose(Image.Transpose.FLIP_LEFT_RIGHT) for im in images]
    arrays = [np.asarray(im, dtype=np.float32)/127.5-1 for im in images]
    x = np.concatenate(arrays, axis=2).transpose(2, 0, 1)
    if rng is not None and cfg["data"]["noise_std"]:
        x = np.clip(x + rng.normal(0, cfg["data"]["noise_std"], x.shape), -1, 1)
    return np.ascontiguousarray(x, dtype=np.float32), flip
