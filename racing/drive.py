"""Explicit simulator entry point for future collection and driving."""
import argparse
import ctypes
from dataclasses import asdict
import json
import logging
import os
from pathlib import Path
import time
import uuid
from .config import load_config, dump_json, sha256
from .control import Command, Controller, clip
from .recording import Recorder
from .transport import RaceClient
from .diagnostics import Diagnostics
from .session import SimulatorSession, ControlSchedule, decode_telemetry


class Keyboard:
    def __init__(self):
        if os.name != "nt": raise RuntimeError("Manual input uses Windows keys; add a controller adapter for other systems")
        self.steering = 0.0

    @staticmethod
    def down(key): return bool(ctypes.windll.user32.GetAsyncKeyState(key) & 0x8000)

    def command(self, cfg, dt):
        r = cfg["runtime"]
        direction = int(self.down(0x27))-int(self.down(0x25))
        if direction: self.steering = clip(self.steering+direction*r["manual_steering_rate"]*dt, -1, 1)
        else: self.steering -= clip(self.steering, -r["manual_return_rate"]*dt, r["manual_return_rate"]*dt)
        if self.down(ord("B")): return Command(self.steering, 0, cfg["control"]["brake_max"])
        return Command(self.steering, r["manual_throttle"] if self.down(0x26) else 0, 0)


def run(a, cfg):
    predictor = None
    if a.mode == "cnn":
        from .inference import Predictor
        predictor = Predictor(a.checkpoint, cfg)  # validate before any connection
    if a.mode == "baseline":
        from .baseline import BaselinePilot
        baseline_pilot = BaselinePilot(cfg)
        from .config import cte_bounds
        cte_center, cte_limit = cte_bounds(cfg, cfg["runtime"]["scene"])
    keyboard = Keyboard() if a.mode == "manual" or a.display else None
    logdir = Path(a.logdir)/uuid.uuid4().hex
    logdir.mkdir(parents=True, exist_ok=False)
    dump_json(logdir/"run.json", {"mode": a.mode, "config": cfg, "checkpoint_sha256": sha256(a.checkpoint) if a.checkpoint else None, "group_id": a.group})
    recorder = client = None
    diagnostics = Diagnostics()
    from .preview import Preview
    preview = Preview(logdir, diagnostics, keyboard is not None, a.snapshots) if keyboard or a.snapshots else None
    controller = Controller(cfg)
    try:
        if a.record: recorder = Recorder(a.sessions, a.group, cfg, a.mode)
        with (logdir/"decisions.jsonl").open("w", encoding="utf-8") as log:
            def write(row): log.write(json.dumps(row, allow_nan=False)+"\n"); log.flush()
            client = RaceClient(cfg, diagnostics)
            session = SimulatorSession(client, cfg)
            started = time.monotonic()
            ready_at = None
            last_decision = started
            import random
            perturb_until, perturb_next, perturb_offset, perturb_rng, pedal_push = None, 0.0, 0.0, random.Random(), False
            schedule = ControlSchedule(cfg["control"]["frequency_hz"])
            sequence = uuid.uuid4().hex
            recorded_frames = 0
            while ready_at is None or time.monotonic()-ready_at < a.seconds:
                now = time.monotonic()
                if keyboard:
                    if keyboard.down(0x1B): break
                observation, events = session.poll()
                for event in events:
                    write({"kind": "event", "event": event})
                schedule.offer(observation)
                observation = schedule.take(time.monotonic())
                if observation is None:
                    time.sleep(.002); continue
                frame, received, image, speed, stamp, telemetry = observation
                if ready_at is None:
                    ready_at = time.monotonic()
                    client.last_control = time.monotonic()
                    client.armed = True
                    last_decision = time.monotonic()-1/cfg["control"]["frequency_hz"]
                    diagnostics.event("DRIVE", "Autopilot ready", mode=a.mode)
                now = time.monotonic()
                dt = now-last_decision
                if not 0 < dt <= cfg["runtime"]["timeout_s"]: raise TimeoutError("Control deadline missed")
                source, prediction, latency, info = a.mode, None, None, {}
                # Autonomous modes cannot be changed by SPACE/arrows, even with --display.
                manual = a.mode == "manual" and keyboard is not None and keyboard.down(0x20)
                recovery = a.mode == "manual" and keyboard is not None and keyboard.down(ord("R"))
                valid = False
                policy_started = time.monotonic()
                if manual:
                    command = keyboard.command(cfg, dt)
                    source = "manual"
                    valid = not keyboard.down(ord("Z"))
                    controller.reset()
                    if predictor: predictor.reset()
                elif a.mode == "manual":
                    command = controller.stop()
                    keyboard.steering = 0.0
                elif a.mode == "baseline":
                    command, info = baseline_pilot.step(image, speed, dt, telemetry)
                    valid = (a.baseline_validated and not info.get("lane_lost", False)
                             and telemetry.get("hit", "none") in ("none", "", None)
                             and abs(float(telemetry.get("cte", float("inf")))-cte_center) <= cte_limit)
                    # Optional baseline.perturb (DART-style noise injection): now and then the
                    # executed steering is pushed off for a moment. The label stays the
                    # teacher's own command, so the CNN sees how to come back to the line.
                    shake = cfg["baseline"].get("perturb")
                    if shake:
                        if perturb_until is None: perturb_until, perturb_next = -1.0, stamp+shake["every_s"]
                        node = telemetry.get("activeNode")
                        def inside(spans): return node is not None and any(lo <= node <= hi for lo, hi in spans)
                        no_wheel = inside(shake.get("skip_nodes", []))  # no wheel push in tight spots
                        no_pedal = inside(shake.get("pedal_skip_nodes", shake.get("skip_nodes", [])))
                        if abs(command.steering) > shake.get("max_teacher_steering", 1.0):
                            perturb_until = min(perturb_until, stamp)  # the teacher is turning hard: stop pushing now
                            perturb_next = max(perturb_next, stamp+.5)
                        if stamp >= perturb_next:
                            # Optional perturb.pedal: some pushes act on the pedal instead of the wheel
                            # (throttle cut or boost), so the network also learns how the teacher
                            # recovers its speed, e.g. pushing harder when it is slow on a hill.
                            pedal_push = "pedal" in shake and (no_wheel or perturb_rng.random() < shake.get("pedal_share", .5))
                            if (no_pedal if pedal_push else no_wheel): perturb_next = stamp+.5
                        if stamp >= perturb_next:
                            length = shake.get("pedal_duration_s", 1.0) if pedal_push else shake["duration_s"]
                            perturb_until, perturb_next = stamp+length, stamp+shake["every_s"]*(.5+perturb_rng.random())
                            perturb_offset = (shake["pedal"] if pedal_push else shake["steering"])*perturb_rng.choice((-1.0, 1.0))
                        if stamp < perturb_until:
                            if pedal_push:
                                pushed = clip(command.throttle+perturb_offset, 0, 1)
                                executed = Command(command.steering, pushed, 0.0 if pushed > 0 else command.brake)
                            else:
                                executed = Command(clip(command.steering+perturb_offset, -1, 1), command.throttle, command.brake)
                            info = {**info, "executed_command": asdict(executed)}
                            recovery = True
                        elif stamp < perturb_until+shake.get("recovery_s", 1.0): recovery = True
                else:
                    prediction, latency = predictor.predict(image, speed, stamp)
                    if prediction is None: command = controller.stop()
                    else: command, info = controller.step(prediction, speed, dt)
                diagnostics.timing("policy", time.monotonic()-policy_started)
                if time.monotonic()-received > cfg["runtime"]["max_frame_age_s"]: raise TimeoutError("Decision exceeded frame age budget")
                sent = client.send_packet((Command(**info["executed_command"]) if "executed_command" in info else command).packet(), control=True)
                diagnostics.timing("decision_age", sent-received)
                record = {"kind": "decision", "sequence_id": sequence, "frame_id": frame, "sim_time": stamp, "received_monotonic": received, "sent_monotonic": sent, "decision_age_s": sent-received, "speed": speed, "source": source, "label_valid": valid, "recovery": recovery, "command": asdict(command), "telemetry": telemetry, "prediction": prediction, "inference_ms": latency, **info}
                write_started = time.monotonic()
                write(record)
                diagnostics.timing("log_write", time.monotonic()-write_started)
                if recorder and (not a.record_limit or recorded_frames < a.record_limit):
                    recorder.submit(image, record)
                    recorded_frames += 1
                if preview: preview.submit(image, record)
                last_decision = now
    except BaseException as e:
        if client: client.emergency(f"{type(e).__name__}: {e}")
        raise
    finally:
        try:
            if client:
                fault = client.fault
                client.close()
                dump_json(logdir/"end.json", {"fault": fault, "stopped_monotonic": time.monotonic()})
        finally:
            diagnostics.save(logdir)
            print("[SUMMARY] " + json.dumps(diagnostics.summary()))
            print(f"Protocol logs: {logdir}")
            if recorder: recorder.close()
            if preview: preview.close()
    print(f"Run saved: {logdir}")


def main():
    p = argparse.ArgumentParser(description="Collect/drive only when you explicitly connect")
    p.add_argument("--config", default="configs/race.json")
    p.add_argument("--mode", choices=("manual", "cnn", "baseline"))
    p.add_argument("--checkpoint")
    p.add_argument("--connect", action="store_true")
    p.add_argument("--record", action="store_true")
    p.add_argument("--record-limit", type=int, default=0, help="Maximum recorded frames; 0 means unlimited (requires --record)")
    p.add_argument("--display", action="store_true")
    p.add_argument("--baseline-validated", action="store_true")
    p.add_argument("--sessions", default="data/sessions")
    p.add_argument("--group", default=None)
    p.add_argument("--logdir", default="runs/driving")
    p.add_argument("--seconds", type=float, default=300)
    p.add_argument("--snapshots", type=float, default=0, help="Seconds between diagnostic images (maximum 120; no dataset)")
    p.add_argument("--debug", action="store_true")
    p.add_argument("--diagnose", action="store_true", help="Verify startup and camera, without accelerating")
    p.add_argument("--frames", type=int, default=1, help="Configured images required by --diagnose")
    p.add_argument("--image-out", help="Optional single diagnostic image, not a dataset")
    a = p.parse_args()
    logging.basicConfig(level=logging.DEBUG if a.debug else logging.INFO, format="%(message)s")
    logging.getLogger("PIL").setLevel(logging.WARNING)
    if not a.connect: p.error("Use --connect only when you intend to run the simulator")
    if a.diagnose:
        if a.record: p.error("--diagnose cannot record a dataset")
        if a.frames <= 0: p.error("--frames must be positive")
        from .diagnose import diagnose
        diagnose(load_config(a.config), a.logdir, a.frames, a.image_out)
        return
    if a.mode is None: p.error("Choose --mode or --diagnose")
    if a.mode == "cnn" and not a.checkpoint: p.error("CNN requires --checkpoint")
    if a.record and not a.group: p.error("Collection requires --group (related sessions share a group)")
    if a.record and a.mode != "baseline": p.error("Dataset recording is restricted to the deterministic baseline")
    if a.record and not a.baseline_validated: p.error("Baseline collection requires --baseline-validated")
    if a.record_limit < 0 or (a.record_limit and not a.record): p.error("--record-limit must be positive and used with --record")
    if a.snapshots < 0: p.error("--snapshots must not be negative")
    if a.seconds <= 0: p.error("--seconds must be positive")
    run(a, load_config(a.config))


if __name__ == "__main__":
    try: main()
    except KeyboardInterrupt:
        logging.info("[STOP] Interrupted by user; shutdown completed")
        raise SystemExit(130)
    except (OSError, ValueError, RuntimeError) as error:
        logging.error("[ERROR] %s", error)
        raise SystemExit(1)
