"""Checkpoint loading and temporal inference, also usable as a DonkeyCar Part."""
from collections import deque
import math
import time
from pathlib import Path
import numpy as np
import torch
from .config import validate_config, contract_hash
from .data import nearest
from .model import RacingCNN
from .preprocess import stack_images
from .control import Controller


def load_checkpoint(path, device="cpu"):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1 or not payload.get("trained") or payload.get("completed_epochs", 0) < 1:
        raise ValueError("Not a trained racing checkpoint")
    if not math.isfinite(float(payload["validation_loss"])):
        raise ValueError("Non-finite checkpoint validation loss")
    cfg = payload["config"]
    validate_config(cfg)
    if contract_hash(cfg) != payload["contract_hash"]: raise ValueError("Corrupted preprocessing contract")
    model = RacingCNN(cfg)
    model.load_state_dict(payload["model_state"], strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()): raise ValueError("Non-finite checkpoint weights")
    model.to(device).eval()
    return model, cfg, payload


class Predictor:
    def __init__(self, path, cfg):
        self.device = torch.device(cfg["runtime"]["inference_device"])
        self.model, stored, _ = load_checkpoint(path, self.device)
        if contract_hash(stored) != contract_hash(cfg): raise ValueError("Live camera/model contract differs from checkpoint")
        self.cfg = cfg
        torch.set_num_threads(cfg["training"]["threads"])
        self.history = deque(maxlen=128)
        # Initialize kernels before connecting; this is not a latency benchmark.
        with torch.inference_mode():
            self.model(torch.zeros(1, 3*cfg["model"]["frames"], cfg["preprocess"]["height"], cfg["preprocess"]["width"], device=self.device), torch.zeros(1,1,device=self.device)).cpu()

    def reset(self): self.history.clear()

    def predict(self, image, speed, sim_time):
        if not math.isfinite(speed) or not 0 <= speed <= self.cfg["model"]["speed_scale"]: raise ValueError("Missing/out-of-range speed")
        if self.history and (sim_time <= self.history[-1][0] or sim_time-self.history[-1][0] > self.cfg["data"]["max_gap_s"]): self.reset()
        self.history.append((sim_time, image.copy()))
        m = self.cfg["model"]
        times = [t for t, _ in self.history]
        indices = [nearest(times, sim_time-k*m["frame_spacing_s"], self.cfg["data"]["time_tolerance_s"]) for k in reversed(range(m["frames"]))]
        if any(j is None for j in indices) or len(set(indices)) != len(indices): return None, None
        start = time.perf_counter()
        x, _ = stack_images([self.history[j][1] for j in indices], self.cfg)
        x = torch.from_numpy(x).unsqueeze(0).to(self.device)
        v = torch.tensor([[speed/m["speed_scale"]]], device=self.device)
        with torch.inference_mode(): pred = self.model(x, v).float().cpu().numpy()[0]
        if not np.isfinite(pred).all(): raise ValueError("Non-finite model prediction")
        return pred.tolist(), (time.perf_counter()-start)*1000


class DonkeyCNNPart:
    """veh.add(part, inputs=['cam/image_array','sim/speed'], outputs=['pilot/angle','pilot/throttle','pilot/brake']).

    DonkeyGym must receive all three outputs. The surrounding Vehicle must supply
    a fresh-frame watchdog; this synchronous Part cannot stop a stalled process.
    """
    def __init__(self, checkpoint, cfg):
        self.predictor, self.controller = Predictor(checkpoint, cfg), Controller(cfg)
        self.last = None

    def run(self, image, speed):
        now = time.monotonic()
        dt = .05 if self.last is None else now-self.last
        self.last = now
        try:
            pred, _ = self.predictor.predict(image, float(speed), now)
            command = self.controller.stop() if pred is None else self.controller.step(pred, float(speed), dt)[0]
        except (ValueError, TypeError):
            self.predictor.reset()
            command = self.controller.stop()
        return command.steering, command.throttle, command.brake
