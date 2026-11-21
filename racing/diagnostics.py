"""Compact protocol diagnostics. Encoded image contents are never logged."""
from collections import Counter, deque
import json
import logging
from pathlib import Path
import threading
import time
from .config import dump_json

logger = logging.getLogger(__name__)


class Diagnostics:
    def __init__(self):
        self.started = time.monotonic()
        self.lock = threading.RLock()
        self.message_types = Counter()
        self.rejections = Counter()
        self.image_sizes = Counter()
        self.first_events = {}
        self.timeline = deque(maxlen=4096)
        self.session_state = "waiting_for_connection"
        self.last_stage = "not_connected"
        self.decoded = 0
        self.protocol_errors = 0
        self.controls_sent = 0
        self.nonzero_throttle_sent = 0
        self.network_errors = []
        self.timings = {}

    def timing(self, name, seconds):
        with self.lock:
            self.timings.setdefault(name, deque(maxlen=30000)).append(seconds)

    def timing_summary(self):
        result = {}
        with self.lock:
            for name, values in self.timings.items():
                ordered = sorted(values)
                if ordered:
                    result[name] = {"n": len(ordered), "mean_ms": 1000*sum(ordered)/len(ordered),
                                    "p95_ms": 1000*ordered[int((len(ordered)-1)*.95)],
                                    "max_ms": 1000*ordered[-1]}
        return result

    def event(self, stage, message, **details):
        now = time.monotonic()-self.started
        with self.lock:
            previous = self.timeline[-1]["elapsed_s"] if self.timeline else 0
            row = {"elapsed_s": round(now,6), "since_previous_event_s": round(now-previous,6),
                   "stage": stage, "message": message, **details}
            self.timeline.append(row)
            self.first_events.setdefault(stage, row["elapsed_s"])
            self.last_stage = stage
        logger.info("[%s] %s%s", stage, message,
                    " " + json.dumps(details, ensure_ascii=False) if details else "")

    def received(self, packet):
        kind = packet["msg_type"]
        with self.lock:
            self.message_types[kind] += 1
            first = self.message_types[kind] == 1
        if first or kind != "telemetry":
            self.event("TELEMETRY" if kind == "telemetry" else "SIM", f"Message received: {kind}",
                       keys=list(packet), image_present=bool(packet.get("image")) if kind == "telemetry" else None)

    def image_decoded(self, size):
        with self.lock:
            self.decoded += 1
            key = f"{size[0]}x{size[1]} RGB"
            first = not self.image_sizes[key]
            self.image_sizes[key] += 1
        if first: self.event("CAMERA", "Image decoded", size=list(size))

    def rejected(self, reason, **details):
        with self.lock:
            self.rejections[reason] += 1
            first = self.rejections[reason] == 1
        if first: self.event("REJECTED", reason, **details)
        else: logger.debug("[REJECTED] %s", reason)

    def network_error(self, reason):
        with self.lock: self.network_errors.append(reason)
        self.event("NETWORK_ERROR", reason)

    def summary(self):
        with self.lock:
            return {"elapsed_s": round(time.monotonic()-self.started,3), "last_stage": self.last_stage,
                    "session_state": self.session_state, "messages_received": sum(self.message_types.values()),
                    "message_types": dict(self.message_types), "telemetry_packets": self.message_types["telemetry"],
                    "images_decoded": self.decoded, "decoded_sizes": dict(self.image_sizes),
                    "images_rejected": sum(self.rejections.values()), "rejection_reasons": dict(self.rejections),
                    "protocol_errors": self.protocol_errors, "controls_sent": self.controls_sent,
                    "nonzero_throttle_sent": self.nonzero_throttle_sent,
                    "network_errors": list(self.network_errors), "first_events_s": dict(self.first_events),
                    "timings": self.timing_summary()}

    def save(self, directory):
        directory = Path(directory)
        dump_json(directory/"protocol_summary.json", self.summary())
        with self.lock: rows = list(self.timeline)
        (directory/"protocol.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in rows), encoding="utf-8")
