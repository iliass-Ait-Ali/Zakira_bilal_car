"""Event-driven menu/track/car/camera startup, independent of the driving policy."""
import base64
import binascii
from io import BytesIO
import math
import time
from PIL import Image, UnidentifiedImageError


class TelemetryError(ValueError):
    def __init__(self, reason, **details):
        self.reason, self.details = reason, details
        super().__init__(f"{reason}: {details}" if details else reason)


class ControlSchedule:
    """Keep the freshest observation until the next control deadline, never a backlog."""
    def __init__(self, frequency_hz):
        self.period = 1/frequency_hz
        self.next_time = 0.0
        self.pending = None

    def offer(self, observation):
        if observation is not None: self.pending = observation

    def take(self, now):
        if self.pending is None or now < self.next_time: return None
        result, self.pending = self.pending, None
        self.next_time = now+self.period
        return result


def decode_telemetry(packet, cfg):
    if not packet.get("image"):
        raise TelemetryError("missing_image")
    encoded = packet["image"]
    if not isinstance(encoded, str): raise TelemetryError("image_not_base64_string")
    if len(encoded) > 8_000_000: raise TelemetryError("oversized_image")
    try: raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise TelemetryError("invalid_base64") from error
    try:
        with Image.open(BytesIO(raw)) as im:
            size, encoding = im.size, im.format
            expected = (cfg["camera"]["img_w"], cfg["camera"]["img_h"])
            if size != expected:
                raise TelemetryError("wrong_resolution", observed=list(size), expected=list(expected))
            expected_format = {"JPG": "JPEG"}.get(cfg["camera"]["img_enc"], cfg["camera"]["img_enc"])
            if encoding != expected_format:
                raise TelemetryError("wrong_encoding", observed=encoding, expected=expected_format)
            image = im.convert("RGB")
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as error:
        raise TelemetryError("undecodable_image", error_type=type(error).__name__) from error
    try: speed, stamp = float(packet["speed"]), float(packet["time"])
    except (KeyError, TypeError, ValueError) as error:
        raise TelemetryError("missing_or_invalid_speed_time") from error
    if not math.isfinite(speed) or speed < 0 or not math.isfinite(stamp):
        raise TelemetryError("nonfinite_or_negative_speed_time")
    telemetry = {k:v for k,v in packet.items() if k not in ("image", "imageb", "lidar")}
    return image, speed, stamp, telemetry


class SimulatorSession:
    def __init__(self, client, cfg):
        self.client, self.cfg = client, cfg
        self.diag = client.diagnostics
        self.started = time.monotonic()
        self.state = "waiting_for_server"
        self.diag.session_state = self.state
        self.scene_requested = False
        self.menu_query_at = None
        self.camera_due = None
        self.camera_sent_at = None
        self.previous_frame = 0
        self.previous_stamp = None
        self.ready = False

    def transition(self, state):
        self.state = state
        self.diag.session_state = state

    def _request_menu(self):
        self.diag.event("SCENE", "Vehicle already loaded; returning to menu to select the requested track")
        self.client.send_packet({"msg_type": "exit_scene"})
        self.transition("waiting_for_menu")
        # This build does not emit scene_selection_ready on every return to menu.
        # A scene_names response confirms the menu handler is installed.
        self.menu_query_at = time.monotonic()+.2

    def on_event(self, packet):
        kind = packet["msg_type"]
        if kind == "aborted": raise ConnectionError(packet.get("reason", "Socket disconnected"))
        if kind == "scene_selection_ready":
            if self.ready: raise RuntimeError("Unexpected return to scene menu")
            if self.state in ("waiting_for_server", "waiting_for_menu"):
                self.transition("waiting_for_scene_names")
                self.client.send_packet({"msg_type": "get_scene_names"})
                self.menu_query_at = time.monotonic()+.5
        elif kind == "scene_names" and self.state in ("waiting_for_menu", "waiting_for_scene_names"):
            names = packet.get("scene_names", [])
            scene = self.cfg["runtime"]["scene"]
            if scene not in names: raise ValueError(f"Requested track {scene!r} not in simulator scene_names: {names}")
            self.diag.event("SCENE", "Loading requested track", scene=scene)
            self.client.send_packet({"msg_type": "load_scene", "scene_name": scene})
            self.scene_requested = True
            self.menu_query_at = None
            self.transition("loading_scene")
        elif kind in ("car_loaded", "need_car_config"):
            if self.ready: raise RuntimeError("Unexpected vehicle reload; start a new session")
            if self.state == "waiting_for_server":
                self._request_menu()
            elif self.state == "loading_scene":
                self.diag.event("CAR_READY", f"Vehicle event observed: {kind}")
                self.client.configure_car()
                self.transition("configuring_car")
                self.camera_due = time.monotonic()+.2
            # Duplicate events during configuration do not resend/reload anything.

    def poll(self):
        now = time.monotonic()
        if self.client.fault or self.client.aborted:
            raise ConnectionError(self.client.fault or "Socket disconnected")
        latest, events = self.client.snapshot()
        for event in events: self.on_event(event)
        if self.state == "waiting_for_server" and latest is not None:
            self._request_menu()  # telemetry also proves a vehicle exists
        if self.menu_query_at is not None and now >= self.menu_query_at:
            self.client.send_packet({"msg_type": "get_scene_names"})
            self.menu_query_at = now+.5
        if self.camera_due is not None and now >= self.camera_due:
            self.camera_sent_at = self.client.configure_camera()
            self.camera_due = None
            self.transition("waiting_for_camera")
        observation = None
        if latest is not None and self.camera_sent_at is not None and latest[0] != self.previous_frame:
            frame, received, packet = latest
            self.previous_frame = frame
            if received >= self.camera_sent_at:
                try:
                    decode_started = time.monotonic()
                    self.diag.timing("age_before_decode", decode_started-received)
                    image, speed, stamp, telemetry = decode_telemetry(packet, self.cfg)
                    self.diag.timing("image_decode", time.monotonic()-decode_started)
                    if time.monotonic()-received > self.cfg["runtime"]["max_frame_age_s"]:
                        raise TelemetryError("stale_frame", age_s=time.monotonic()-received,
                                             decode_s=time.monotonic()-decode_started)
                    if self.previous_stamp is not None and stamp <= self.previous_stamp:
                        raise TelemetryError("simulator_time_stopped_or_reset")
                except TelemetryError as error:
                    self.diag.rejected(error.reason, **error.details)
                    if self.ready: raise
                else:
                    self.previous_stamp = stamp
                    self.diag.image_decoded(image.size)
                    if not self.ready:
                        self.ready = True
                        self.transition("ready")
                        self.diag.event("READY", "Configured camera telemetry is ready", scene=self.cfg["runtime"]["scene"])
                    observation = (frame, received, image, speed, stamp, telemetry)
        if not self.ready and now-self.started > self.cfg["runtime"]["startup_timeout_s"]:
            summary = self.diag.summary()
            raise TimeoutError(f"Startup stopped at {self.state}; messages={summary['message_types']}; "
                               f"telemetry={summary['telemetry_packets']}; rejected={summary['rejection_reasons']}")
        return observation, events
