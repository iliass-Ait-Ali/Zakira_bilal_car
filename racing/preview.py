"""Optional GUI and sparse diagnostic images, never on the control thread."""
import threading
import time


class Preview:
    def __init__(self, directory, diagnostics, display=False, snapshot_period=0):
        self.directory, self.diag = directory, diagnostics
        self.display, self.period = display, snapshot_period
        self.lock = threading.Lock()
        self.pending = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="racing-preview")
        self.thread.start()

    def submit(self, image, record):
        with self.lock: self.pending = (image, record)

    def _run(self):
        if self.display:
            import cv2
            import numpy as np
        next_snapshot, count = 0, 0
        try:
            while not self.stop.is_set():
                with self.lock: item, self.pending = self.pending, None
                if item is not None:
                    image, row = item
                    started = time.monotonic()
                    if self.period and row["sim_time"] >= next_snapshot and count < 120:
                        path = self.directory/"snapshots"
                        path.mkdir(exist_ok=True)
                        image.save(path/f"{row['sim_time']:09.3f}_node{row['telemetry'].get('activeNode', -1):03}.png")
                        next_snapshot, count = row["sim_time"]+self.period, count+1
                    if self.display:
                        frame = cv2.resize(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR), (510,510))
                        cv2.putText(frame, f"{row['source']} v={row['speed']:.2f} s={row['command']['steering']:.2f}",
                                    (8,25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,255,0), 1)
                        cv2.putText(frame, f"node {row['telemetry'].get('activeNode')} | ESC: stop", (8,490),
                                    cv2.FONT_HERSHEY_SIMPLEX, .55, (0,255,0), 1)
                        cv2.imshow("Racing camera", frame)
                    self.diag.timing("preview_render_and_snapshot", time.monotonic()-started)
                if self.display:
                    started = time.monotonic()
                    cv2.waitKey(1)
                    self.diag.timing("display_wait", time.monotonic()-started)
                self.stop.wait(.01)
        except Exception as error:
            self.diag.event("PREVIEW_ERROR", str(error))
        finally:
            if self.display: cv2.destroyAllWindows()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=3)
