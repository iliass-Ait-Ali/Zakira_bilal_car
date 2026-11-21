"""Limited real connection/image test. No dataset and no nonzero throttle."""
import json
from pathlib import Path
import time
import uuid
from .config import dump_json
from .control import Command
from .diagnostics import Diagnostics
from .session import SimulatorSession
from .transport import RaceClient


def diagnose(cfg, logroot="runs/diagnostics", frames=1, image_out=None):
    directory = Path(logroot)/("diagnose_"+uuid.uuid4().hex)
    directory.mkdir(parents=True, exist_ok=False)
    diagnostics = Diagnostics()
    dump_json(directory/"run.json", {"mode": "diagnose", "config": cfg, "dataset_recording": False})
    client = None
    count = 0
    failure = None
    try:
        client = RaceClient(cfg, diagnostics)
        session = SimulatorSession(client, cfg)
        deadline = time.monotonic()+cfg["runtime"]["startup_timeout_s"]
        while count < frames:
            observation, _ = session.poll()
            if observation is not None:
                _, _, image, _, _, _ = observation
                count += 1
                if count == 1:
                    client.send_packet(Command(0, 0, cfg["control"]["emergency_brake"]).packet(), control=True)
                    if image_out:
                        output = Path(image_out)
                        output.parent.mkdir(parents=True, exist_ok=True)
                        image.save(output)
                        diagnostics.event("IMAGE_SAVED", "Saved one diagnostic image", path=str(output))
            if time.monotonic() > deadline and count < frames:
                raise TimeoutError(f"Received {count}/{frames} configured images within the diagnostic deadline")
            time.sleep(.002)
        diagnostics.event("DIAGNOSE_OK", "Real camera reception verified; no acceleration", frames=count)
    except BaseException as error:
        failure = f"{type(error).__name__}: {error}"
        diagnostics.event("FAILED", failure)
        if client: client.emergency(failure)
        raise
    finally:
        try:
            if client: client.close()
        finally:
            dump_json(directory/"result.json", {"success": failure is None and count >= frames,
                                              "verified_images": count, "failure": failure})
            diagnostics.save(directory)
            print("[SUMMARY] " + json.dumps(diagnostics.summary()))
            print(f"Protocol logs: {directory}")
    return directory
