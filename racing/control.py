"""Dimensionless steering and speed control in simulator telemetry units."""
from dataclasses import dataclass, asdict
import math


def clip(value, lo, hi):
    return min(hi, max(lo, value))


@dataclass(frozen=True)
class Command:
    steering: float
    throttle: float
    brake: float

    def __post_init__(self):
        for name, lo, hi in (("steering", -1, 1), ("throttle", 0, 1), ("brake", 0, 1)):
            v = getattr(self, name)
            if not math.isfinite(v) or not lo <= v <= hi:
                raise ValueError(f"Invalid command {name}: {v}")
        if self.throttle > 0 and self.brake > 0:
            raise ValueError("Throttle and brake are mutually exclusive")

    def packet(self):
        return {"msg_type": "control", **{k: str(v) for k, v in asdict(self).items()}}


class Controller:
    def __init__(self, cfg):
        self.c = cfg["control"]
        self.horizons = len(cfg["model"]["horizons_s"])
        self.pedal = bool(cfg["model"].get("throttle_output", False))
        # control.throttle_source = "network": throttle and brake come straight from the
        # network's pedal output (fully learned driving). Default "rule": the speed target
        # below, computed from the predicted steering, with a PI throttle.
        self.network_pedal = self.c.get("throttle_source", "rule") == "network"
        if self.network_pedal and not self.pedal: raise ValueError("throttle_source=network needs model.throttle_output")
        self.reset()

    def reset(self):
        self.steer = 0.0
        self.integral = 0.0
        self.braking = False
        self.target = self.c["speed_turn"]

    def stop(self):
        self.reset()
        return Command(0.0, 0.0, self.c["emergency_brake"])

    def step(self, prediction, speed, dt):
        if len(prediction) != self.horizons+self.pedal or not all(math.isfinite(float(v)) and abs(v) <= 1.001 for v in prediction):
            raise ValueError("Invalid CNN prediction")
        if not math.isfinite(speed) or speed < 0 or not 0 < dt <= .5:
            raise ValueError("Invalid speed or control interval")
        c = self.c
        raw = clip(float(prediction[0]), -1, 1)
        alpha = 1.0 if c["steering_tau_s"] == 0 else 1-math.exp(-dt/c["steering_tau_s"])
        self.steer += clip(alpha*(raw-self.steer), -c["steering_rate"]*dt, c["steering_rate"]*dt)
        # Future steering is a turn-demand proxy, not a measured curvature.
        demand = max(abs(float(v)) for v in prediction[:self.horizons])
        if self.network_pedal:
            pedal = float(prediction[self.horizons])
            throttle = clip(pedal, 0, c.get("network_throttle_max", 1.0)) if pedal > 0 else 0.0
            brake = clip(-pedal, 0, c["brake_max"]) if pedal < -c.get("network_brake_deadband", .05) else 0.0
            self.target = speed
            return Command(self.steer, throttle, brake), {"target_speed": speed, "turn_demand": demand, "pedal": pedal}
        target = max(c["speed_turn"], c["speed_straight"]/(1+c["turn_gain"]*demand))
        self.target = min(target, self.target+c["speed_rise_per_s"]*dt)
        error = self.target-speed
        if error < -c["brake_enter_error"]: self.braking = True
        elif error > -c["brake_exit_error"]: self.braking = False
        if self.braking:
            self.integral = 0.0
            throttle, brake = 0.0, clip(-error*c["brake_kp"], 0, c["brake_max"])
        elif error <= 0:
            self.integral = 0.0
            throttle, brake = 0.0, 0.0
        else:
            proposed = clip(self.integral+error*dt, -c["integral_limit"], c["integral_limit"])
            effort = c["throttle_ff"]+c["throttle_kp"]*error+c["throttle_ki"]*proposed
            if effort <= c["throttle_max"]: self.integral = proposed
            throttle, brake = clip(effort, 0, c["throttle_max"]), 0.0
        return Command(self.steer, throttle, brake), {"target_speed": self.target, "turn_demand": demand}
