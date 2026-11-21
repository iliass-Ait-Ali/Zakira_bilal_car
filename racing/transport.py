"""SDSandbox transport adapter; no inference or image decoding in callbacks."""
from collections import deque
import json
import logging
import threading
import time
from .diagnostics import Diagnostics
from .vendor.client import SDClient

logger = logging.getLogger(__name__)


def encode_packet(packet):
    # Server output is newline-delimited. This build's input parser instead
    # requires each JSON object's LAST character to be }, with no newline.
    value = json.dumps(packet, allow_nan=False, separators=(",", ":"))
    if not value.startswith("{") or not value.endswith("}"):
        raise ValueError("Outgoing packet must be a JSON object")
    return value


class RaceClient(SDClient):
    def __init__(self, cfg, diagnostics=None):
        self.cfg = cfg
        self.diagnostics = diagnostics or Diagnostics()
        self.condition = threading.Condition()
        self.send_lock = threading.Lock()
        self.latest = None
        self.counter = 0
        self.events = deque()
        self.fault = None
        self.last_rx = self.last_control = time.monotonic()
        self.armed = False
        self.closed = False
        self.watch_stop = threading.Event()
        self.diagnostics.event("CONNECT", "Connecting", host=cfg["runtime"]["host"], port=cfg["runtime"]["port"])
        try:
            super().__init__(cfg["runtime"]["host"], cfg["runtime"]["port"],
                             poll_socket_sleep_time=cfg["runtime"]["poll_s"],
                             send_timeout=cfg["runtime"]["timeout_s"])
        except OSError as error:
            self.diagnostics.network_error(f"TCP connection failed: {error}")
            raise
        self.watch = threading.Thread(target=self._watchdog, daemon=True, name="racing-watchdog")
        self.watch.start()

    def on_connected(self):
        self.diagnostics.event("CONNECTED", "TCP connection established")

    def on_msg_recv(self, packet):
        now = getattr(self, "last_message_received_at", time.monotonic())
        self.diagnostics.received(packet)
        with self.condition:
            kind = packet["msg_type"]
            if kind == "telemetry":
                if self.counter: self.diagnostics.timing("receive_interval", now-self.last_rx)
                self.diagnostics.timing("json_decode", getattr(self, "last_json_decode_s", 0))
                self.counter += 1
                self.latest = (self.counter, now, packet)
                self.last_rx = now
                if packet.get("hit", "none") not in ("none", "", None):
                    self.events.append({"msg_type": "hit", "hit": packet["hit"], "time": packet.get("time"), "received_monotonic": now})
            else:
                self.events.append(dict(packet, received_monotonic=now))
            if kind == "aborted":
                self.fault = packet.get("reason", "Socket aborted")
                self.diagnostics.network_error(self.fault)
            if len(self.events) > 4096: self.fault = "event_queue_overflow"
            self.condition.notify_all()

    def on_protocol_error(self, reason):
        with self.diagnostics.lock: self.diagnostics.protocol_errors += 1
        self.diagnostics.event("PROTOCOL_ERROR", reason)

    def send_packet(self, packet, control=False):
        with self.send_lock:
            if self.closed or self.fault or self.aborted:
                raise ConnectionError(self.fault or "Client is closed/disconnected")
            try: super().send_now(encode_packet(packet))
            except (OSError, ValueError) as error:
                self.fault = f"TCP send failed: {error}"
                self.diagnostics.network_error(self.fault)
                raise ConnectionError(self.fault) from error
            now = time.monotonic()
            if control:
                self.last_control = now
                with self.diagnostics.lock:
                    self.diagnostics.controls_sent += 1
                    self.diagnostics.nonzero_throttle_sent += float(packet["throttle"]) > 0
            else:
                logger.debug("[SEND] %s", encode_packet(packet))
            return now

    def emergency(self, reason):
        with self.send_lock:
            if self.closed: return
            self.fault = self.fault or reason
            self.armed = False
            try:
                super().send_now(encode_packet({"msg_type": "control", "steering": "0", "throttle": "0",
                                               "brake": str(self.cfg["control"]["emergency_brake"])}))
                self.diagnostics.event("STOP", "Stop command sent", reason=reason)
            except (OSError, ValueError) as error:
                self.diagnostics.event("STOP_FAILED", "Stop could not be transmitted", reason=str(error))

    def _watchdog(self):
        while not self.watch_stop.wait(.02):
            if self.armed and not self.fault:
                now = time.monotonic()
                if max(now-self.last_rx, now-self.last_control) > self.cfg["runtime"]["timeout_s"]:
                    self.emergency("watchdog_timeout")

    def snapshot(self):
        with self.condition:
            events = list(self.events)
            self.events.clear()
            return self.latest, events

    def configure_car(self):
        self.diagnostics.event("CAR", "Sending car configuration")
        # racer_info is absent from this build's dispatcher (confirmed in Player.log).
        self.send_packet({"msg_type": "car_config", "body_style": "car01", "body_r": "255", "body_g": "0", "body_b": "255", "car_name": "RaceCNN", "font_size": "100"})

    def configure_camera(self):
        # rot_x is mandatory in this build. Zero preserves the prefab rotation.
        camera = {"rot_x": 0, **self.cfg["camera"]}
        self.diagnostics.event("CAMERA_CONFIG", "Sending camera configuration", camera=camera)
        sent = self.send_packet({"msg_type": "cam_config", **{k: str(v) for k,v in camera.items()}})
        # {enabled:false} is unsupported for cam_config_b and lidar_config in this
        # binary. Leave optional sensors at their defaults and ignore their output.
        return sent

    def configure(self):
        """Compatibility entry point; normal startup uses timed Session steps."""
        self.configure_car()
        time.sleep(.2)
        return self.configure_camera()

    def close(self):
        if self.closed: return
        self.armed = False
        self.watch_stop.set()
        self.watch.join(timeout=1)
        if not self.fault: self.emergency("normal_shutdown")
        super().stop()  # join receiver BEFORE closing its descriptor
        self.closed = True
        self.diagnostics.event("DISCONNECTED", "Receiver stopped and socket closed")
