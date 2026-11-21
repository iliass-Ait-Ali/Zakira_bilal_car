"""Bounded background image writer. Queue overflow aborts, never mislabels frames."""
import json
import queue
import threading
import uuid
from pathlib import Path
from .config import dump_json


class Recorder:
    def __init__(self, root, group, cfg, source):
        self.session_id = uuid.uuid4().hex
        self.path = Path(root)/self.session_id
        (self.path/"images").mkdir(parents=True, exist_ok=False)
        dump_json(self.path/"session.json", {"schema_version": 1, "session_id": self.session_id, "group_id": group, "camera": cfg["camera"], "source": source, "config": cfg, "alignment": "decision from this received image; sent command is not an acknowledged application timestamp"})
        self.queue = queue.Queue(maxsize=128)
        self.error = None
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def submit(self, image, record):
        if self.error: raise RuntimeError("Dataset writer failed") from self.error
        record = dict(record, session_id=self.session_id)
        try: self.queue.put_nowait((image.copy(), record))
        except queue.Full as e: raise RuntimeError("Dataset writer too slow; stop collection") from e

    def _write(self):
        try:
            with (self.path/"records.jsonl").open("w", encoding="utf-8") as f:
                index = 0
                while True:
                    item = self.queue.get()
                    if item is None: break
                    image, record = item
                    name = f"images/{index:08d}.png"
                    image.save(self.path/name)
                    record["image"] = name
                    f.write(json.dumps(record, allow_nan=False)+"\n"); f.flush()
                    index += 1
        except Exception as e: self.error = e

    def close(self):
        while self.thread.is_alive():
            try: self.queue.put(None, timeout=.1); break
            except queue.Full: continue
        self.thread.join(timeout=10)
        if self.thread.is_alive(): raise RuntimeError("Writer did not finish within 10 seconds")
        if self.error: raise RuntimeError("Dataset writer failed") from self.error
