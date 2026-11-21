"""Shared configuration and data contract, embedded in every trained checkpoint."""
import hashlib
import json
import math
from pathlib import Path


def dump_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def load_config(path="configs/race.json"):
    cfg = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    validate_config(cfg)
    return cfg


def validate_config(c):
    def require(ok, message):
        if not ok:
            raise ValueError(message)
    def finite(v):
        if isinstance(v, dict):
            for x in v.values(): finite(x)
        elif isinstance(v, list):
            for x in v: finite(x)
        elif isinstance(v, (float, int)):
            require(math.isfinite(v), "Non-finite configuration")
    finite(c)
    require(c["schema_version"] == 1, "Unsupported configuration version")
    p, m, d, t, ctl, r = (c[k] for k in ("preprocess", "model", "data", "training", "control", "runtime"))
    require(0 <= p["crop_top"] < 1 and 0 <= p["crop_bottom"] < 1 and p["crop_top"] + p["crop_bottom"] < .85, "Invalid crop")
    require(p["width"] >= 64 and p["height"] >= 48, "Input is too small")
    require(p["width"] % 40 == 0 and p["height"] % 24 == 0, "Use width multiple of 40 and height multiple of 24 for spatial pooling")
    temporal = m.get("temporal", "stack")
    require(temporal in ("stack", "lstm"), "model.temporal must be 'stack' or 'lstm'")
    allowed_frames = (1, 3) if temporal == "stack" else tuple(range(2, 9))
    require(m["frames"] in allowed_frames and m["frame_spacing_s"] > 0 and m["speed_scale"] > 0, "Invalid temporal model")
    if temporal == "lstm": require(m.get("lstm_hidden", 128) > 0, "Invalid LSTM size")
    require(len(m["channels"]) == 5 and all(x > 0 for x in m["channels"] + m["dense"]), "Invalid layers")
    require(0 <= m["dropout"] < 1, "Invalid dropout")
    h = m["horizons_s"]
    require(h[0] == 0 and h == sorted(set(h)), "Horizons must start at zero and increase")
    # Optional model.throttle_output: one more network output, the pedal (throttle - brake), with its own loss weight.
    require(isinstance(m.get("throttle_output", False), bool), "model.throttle_output must be true or false")
    require(len(h)+int(m.get("throttle_output", False)) == len(t["loss_weights"]) and all(x > 0 for x in t["loss_weights"]), "Invalid loss weights")
    require(0 < d["val_fraction"] < 1 and 0 < d["test_fraction"] < 1 and d["val_fraction"] + d["test_fraction"] < 1, "Invalid split")
    require(d["min_groups"] >= 3 and d["max_gap_s"] > 0 and d["time_tolerance_s"] > 0, "Invalid sequence policy")
    require(0 <= d["flip_probability"] <= 1 and d["balance_power"] >= 0 and d["max_sample_weight"] >= 1, "Invalid sampling")
    require(t["batch_size"] > 0 and t["epochs"] > 0 and t["workers"] >= 0 and t["threads"] > 0, "Invalid training sizes")
    require(0 < t["min_learning_rate"] <= t["learning_rate"] and t["huber_beta"] > 0 and t["patience"] > 0, "Invalid optimizer")
    require(0 < ctl["speed_turn"] <= ctl["speed_straight"] <= m["speed_scale"], "Invalid speed range")
    require(0 < ctl["frequency_hz"] <= 100 and ctl["steering_rate"] > 0 and ctl["steering_tau_s"] >= 0, "Invalid steering controller")
    require(0 <= ctl["brake_exit_error"] < ctl["brake_enter_error"], "Invalid brake hysteresis")
    for k in ("throttle_max", "brake_max", "emergency_brake"):
        require(0 <= ctl[k] <= 1, f"Invalid {k}")
    for k in ("turn_gain", "speed_rise_per_s", "throttle_kp", "throttle_ki", "integral_limit", "throttle_ff", "brake_kp"):
        require(ctl[k] >= 0, f"Invalid {k}")
    require(0 < r["max_frame_age_s"] < r["timeout_s"] < r["startup_timeout_s"], "Invalid watchdog timeouts")
    require(c["camera"]["img_d"] == 3 and c["camera"]["img_w"] > 0 and c["camera"]["img_h"] > 0, "Invalid camera")
    if "baseline" in c:
        b = c["baseline"]
        require(0 < b["speed_turn"] <= b["speed_straight"], "Invalid baseline speeds")
        require(b["gain"] > 0 and b["steering_tau_s"] > 0 and b["steering_rate"] > 0, "Invalid baseline steering")
        require(0 < b["throttle_max"] <= 1 and 0 <= b["throttle_ff"] <= b["throttle_max"], "Invalid baseline throttle")
        require(b["speed_kp"] >= 0 and b["speed_ki"] >= 0, "Invalid baseline speed feedback")


def cte_bounds(c, scene):
    """(centre, max_abs) CTE acceptance for a scene. Tracks whose reference path is not
    the lane centre (e.g. warren, about -2.1) declare it in data.cte_reference."""
    ref = c["data"].get("cte_reference", {}).get(scene, {})
    return float(ref.get("center", 0.0)), float(ref.get("max_abs", c["data"]["max_abs_cte"]))


def data_scenes(c):
    return c["data"].get("scenes", [c["runtime"]["scene"]])


def contract(c):
    return {k: c[k] for k in ("schema_version", "camera", "preprocess", "model")}


def contract_hash(c):
    return hashlib.sha256(json.dumps(contract(c), sort_keys=True).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""): h.update(block)
    return h.hexdigest()
