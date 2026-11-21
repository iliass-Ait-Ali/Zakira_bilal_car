"""Rebuild the mission evidence from actual logs; does not connect or train."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from racing.config import dump_json

RUNS = ["b7b4805b14e046918a2a24401a88929a", "7d95624cf6e44c9ab38574684d07b1f8",
        "2dbda007dd8c46dba65ca0c3a4cd86ba", "885e5dd421a94306ae7722714fe75a16"]


def main():
    summary = []
    for identifier in RUNS:
        directory = ROOT/"runs/lap_checks"/identifier
        protocol = json.loads((directory/"protocol_summary.json").read_text())
        laps = json.loads((directory/"lap_validation.json").read_text())
        rows = [json.loads(s) for s in (directory/"decisions.jsonl").read_text().splitlines()]
        decisions = [r for r in rows if r["kind"] == "decision"]
        summary.append({"run":identifier, "drive_wall_s":protocol["first_events_s"]["STOP"]-protocol["first_events_s"]["DRIVE"],
                        "laps":laps, "protocol":protocol,
                        "collision_frames":sum(r["telemetry"].get("hit") not in ("none","",None) for r in decisions),
                        "diagnostic_snapshots":len(list((directory/"snapshots").glob("*.png"))),
                        "decisions_sha256":hashlib.sha256((directory/"decisions.jsonl").read_bytes()).hexdigest()})
    recording = json.loads((ROOT/"runs/expert_recording_check/verification.json").read_text())
    dump_json(ROOT/"docs/lap_evidence.json", {"simulator_scene":"roboracingleague_1", "attempts":len(summary),
              "validated_laps":sum(r["laps"]["validated_laps"] for r in summary), "runs":summary,
              "recording_check":recording, "training_started":False})

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    directory=ROOT/"runs/lap_checks"/RUNS[-1]
    files=sorted((directory/"snapshots").glob("*.png"))
    targets=[7,14,21,30,38,45,51,57,86,114,142,170]
    fig,axes=plt.subplots(3,4,figsize=(12,10))
    for ax,target in zip(axes.flat,targets):
        path=min(files,key=lambda p:abs(float(p.stem.split("_")[0])-target))
        ax.imshow(Image.open(path)); ax.axis("off")
        stamp,node=path.stem.split("_")
        ax.set_title(f"t={float(stamp):.2f} s — {node}",fontsize=10)
    fig.suptitle("Captures réelles de la caméra — essai autonome de 180 s",fontsize=15)
    fig.tight_layout(); fig.savefig(directory/"camera_evidence.png",dpi=130); plt.close(fig)
    print("Evidence rebuilt: 4 attempts,",sum(r["laps"]["validated_laps"] for r in summary),"validated laps")


if __name__ == "__main__": main()
