"""Session-safe supervised records; invalid records break temporal sequences."""
import argparse
import bisect
import json
import math
import hashlib
import logging
from io import BytesIO
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from .config import load_config, dump_json, contract_hash, sha256, cte_bounds, data_scenes
from .control import Command


def require_baseline_policy(cfg):
    if cfg["data"]["allowed_sources"] != ["baseline"]:
        raise ValueError("This project permits baseline-only datasets; set allowed_sources to ['baseline']")


def checked_record(record, meta, session_path, cfg):
    """Validate provenance, safety, alignment and actual pixels, without changing originals."""
    r = dict(record)
    if meta.get("source") != "baseline" or r.get("source") != "baseline":
        raise ValueError("Source is not baseline")
    if r["session_id"] != meta["session_id"]: raise ValueError("Session mismatch")
    if r.get("label_valid") is not True:
        # Optional (data.revalidate_scenes): the recorder applied the default CTE bound,
        # which does not fit tracks whose reference path is not the lane centre. For those
        # scenes the same criteria (no hit, lane seen, track-specific CTE bound) are
        # re-applied below; anything else stays rejected.
        scene = meta.get("config", {}).get("runtime", {}).get("scene")
        if not (scene in cfg["data"].get("revalidate_scenes", []) and r.get("source") == "baseline"):
            raise ValueError("Record is not an expert demonstration")
    if not isinstance(r["sequence_id"], str) or not r["sequence_id"]: raise ValueError("Missing sequence")
    if type(r["frame_id"]) is not int or r["frame_id"] < 1: raise ValueError("Invalid frame id")
    for key in ("sim_time", "received_monotonic", "sent_monotonic", "decision_age_s", "speed"):
        if not math.isfinite(float(r[key])): raise ValueError(f"Non-finite {key}")
    if r["sim_time"] < 0: raise ValueError("Invalid simulator time")
    age = r["sent_monotonic"]-r["received_monotonic"]
    if age < 0 or not math.isclose(age, r["decision_age_s"], abs_tol=1e-6): raise ValueError("Inconsistent decision clocks")
    if not 0 <= age <= cfg["data"]["max_decision_age_s"]: raise ValueError("Stale decision")
    if not 0 <= r["speed"] <= cfg["model"]["speed_scale"]: raise ValueError("Invalid speed")
    Command(**r["command"])
    tel = r["telemetry"]
    if tel["hit"] not in ("none", "", None): raise ValueError("Collision frame")
    center, limit = cte_bounds(cfg, meta.get("config", {}).get("runtime", {}).get("scene", cfg["runtime"]["scene"]))
    if not math.isfinite(float(tel["cte"])) or abs(tel["cte"]-center) > limit:
        raise ValueError("CTE exceeds expert quality limit")
    if r.get("lane_lost", False) or r["lane_confidence"] < 4: raise ValueError("Lane detection lost")
    if not math.isclose(float(tel["time"]), r["sim_time"], abs_tol=1e-6): raise ValueError("Telemetry time mismatch")
    if not math.isclose(float(tel["speed"]), r["speed"], abs_tol=1e-6): raise ValueError("Telemetry speed mismatch")
    image = (session_path/r["image"]).resolve()
    if not image.is_relative_to(session_path.resolve()): raise ValueError("Image outside session")
    raw = image.read_bytes()
    with Image.open(BytesIO(raw)) as im:
        im.load()
        if im.size != (cfg["camera"]["img_w"], cfg["camera"]["img_h"]) or im.mode != "RGB" or im.format != "PNG":
            raise ValueError("Unexpected camera resolution, color mode or encoding")
        pixel_hash = hashlib.sha256(im.tobytes()).hexdigest()
    r.update(image=str(image), image_sha256=hashlib.sha256(raw).hexdigest(), pixel_sha256=pixel_hash)
    return r


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def nearest(times, target, tolerance):
    j = bisect.bisect_left(times, target)
    candidates = [i for i in (j-1, j) if 0 <= i < len(times)]
    if not candidates: return None
    i = min(candidates, key=lambda k: abs(times[k]-target))
    return i if abs(times[i]-target) <= tolerance else None


def sequence_samples(rows, cfg):
    """Rows belong to one uninterrupted session segment, ordered in sim time."""
    times = [r["sim_time"] for r in rows]
    m, d = cfg["model"], cfg["data"]
    output = []
    for i, row in enumerate(rows):
        history = [nearest(times, times[i]-k*m["frame_spacing_s"], d["time_tolerance_s"]) for k in reversed(range(m["frames"]))]
        if any(j is None or j > i for j in history) or len(set(history)) != len(history): continue
        indices = [nearest(times, times[i]+h, d["time_tolerance_s"]) for h in m["horizons_s"]]
        # Never manufacture a future label at a session end or across a gap.
        labels = [float(rows[j]["command"]["steering"]) if j is not None and j >= i else 0.0 for j in indices]
        mask = [float(j is not None and j >= i and (h == 0 or j > i)) for j, h in zip(indices, m["horizons_s"])]
        output.append({"session_id": row["session_id"], "group_id": row["group_id"], "sequence_id": row["sequence_id"],
                       "source": row["source"], "sim_time": times[i], "images": [rows[j]["image"] for j in history],
                       "image_sha256": [rows[j].get("image_sha256") for j in history],
                       "pixel_sha256": [rows[j].get("pixel_sha256") for j in history],
                       "steering": labels, "mask": mask, "speed": row["speed"], "recovery": row.get("recovery", False),
                       "pedal": max(-1.0, min(1.0, float(row["command"].get("throttle", 0.0))-float(row["command"].get("brake", 0.0))))})
    return output


def load_sessions(root, cfg):
    require_baseline_policy(cfg)
    samples, rejected, summaries, segments = [], [], [], []
    files = sorted(Path(root).glob("*/records.jsonl"))
    if not files:
        raise ValueError("No supervised sessions found; src/camera images have no labels. Collect data first.")
    seen_ids = set()
    sequence_groups = {}
    for path in files:
        meta = json.loads(path.with_name("session.json").read_text(encoding="utf-8"))
        sid = meta["session_id"]
        if sid in seen_ids: raise ValueError(f"Duplicate session id: {sid}")
        seen_ids.add(sid)
        if meta["camera"] != cfg["camera"]:
            raise ValueError(f"Camera configuration mismatch: {sid}")
        group = meta["group_id"]
        if not isinstance(group, str) or not group: raise ValueError("Missing independent group id")
        if meta["config"]["runtime"]["scene"] not in data_scenes(cfg): raise ValueError("Simulator scene not listed in data.scenes")
        logging.info("Checking session %s (%s)", sid, group)
        segment, number, last_time, last_sequence = [], 0, None, None
        previous_by_sequence = {}
        def flush():
            nonlocal segment, number
            if segment: segments.append(segment)
            segment = []
            number += 1
        accepted = 0
        total = 0
        with path.open(encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                total += 1
                try:
                    r = checked_record(json.loads(line), meta, path.parent, cfg)
                    t = float(r["sim_time"])
                    sequence = r["sequence_id"]
                    if sequence in sequence_groups and sequence_groups[sequence] != group:
                        raise RuntimeError("Same physical sequence assigned to different groups")
                    sequence_groups[sequence] = group
                    previous = previous_by_sequence.get(sequence)
                    if previous and (t <= previous[0] or r["frame_id"] <= previous[1] or r["received_monotonic"] <= previous[2]):
                        raise ValueError("Non-increasing time or repeated frame")
                    previous_by_sequence[sequence] = (t, r["frame_id"], r["received_monotonic"])
                    if last_time is not None and (t-last_time > cfg["data"]["max_gap_s"] or sequence != last_sequence): flush()
                    last_time, last_sequence = t, sequence
                    speed = float(r["speed"])
                    r.update(sim_time=t, speed=speed, group_id=group, sequence_id=f"{sequence}:{number}", _line=line_num)
                    segment.append(r)
                    accepted += 1
                except (ValueError, TypeError, KeyError, OSError) as e:
                    rejected.append({"session": sid, "line": line_num, "reason": str(e)})
                    # Optional (data.trim_before_invalid_s): the expert may already be
                    # drifting just before it loses the line or leaves the CTE bound.
                    trim = cfg["data"].get("trim_before_invalid_s", 0.0)
                    if trim and segment:
                        cut = segment[-1]["sim_time"]-trim
                        while segment and segment[-1]["sim_time"] > cut:
                            dropped = segment.pop(); accepted -= 1
                            rejected.append({"session": sid, "line": dropped["_line"], "reason": f"Within {trim} s before an invalid frame"})
                    flush()
                    last_time = last_sequence = None
        flush()
        summaries.append({"session": sid, "group": group, "source": meta.get("source"), "records": total,
                          "accepted": accepted, "rejected": total-accepted, "sequences": list(previous_by_sequence),
                          "records_sha256": sha256(path), "session_sha256": sha256(path.with_name("session.json"))})
    # Identical rendered frames can recur at a deterministic restart. Remove them
    # from EVERY involved group before constructing targets; never leak pixels
    # across train/val/test or bridge an excluded frame with future labels.
    pixel_groups = defaultdict(set)
    for segment in segments:
        for r in segment: pixel_groups[r["pixel_sha256"]].add(r["group_id"])
    shared = {digest for digest,groups in pixel_groups.items() if len(groups)>1}
    summary_by_id = {s["session"]:s for s in summaries}
    for segment in segments:
        kept, chunk = [], 0
        for r in segment:
            if r["pixel_sha256"] in shared:
                samples.extend(sequence_samples(kept,cfg)); kept=[]; chunk+=1
                rejected.append({"session":r["session_id"],"line":r["_line"],"reason":"Identical image shared by independent groups"})
                summary_by_id[r["session_id"]]["accepted"]-=1
                summary_by_id[r["session_id"]]["rejected"]+=1
            else:
                r["sequence_id"]+=f":dedup{chunk}"
                kept.append(r)
        samples.extend(sequence_samples(kept,cfg))
    return samples, rejected, summaries


def split_groups(samples, cfg):
    d = cfg["data"]
    groups = sorted({r["group_id"] for r in samples})
    if len(groups) < d["min_groups"]:
        raise ValueError(f"Need {d['min_groups']} independent drive groups, found {len(groups)}. Related sessions must share group_id.")
    if "val_groups" in d or "test_groups" in d:
        # Optional explicit split (whole groups only), e.g. one session of every track in
        # validation; listed groups that have no accepted sample are simply absent.
        val, test = set(d.get("val_groups", [])), set(d.get("test_groups", []))
        if val & test: raise ValueError("A group cannot be in validation and test")
        mapping = {g: "val" if g in val else "test" if g in test else "train" for g in groups}
        if not all(v in mapping.values() for v in ("train", "val", "test")): raise ValueError("Explicit split leaves an empty set")
    else:
        np.random.default_rng(d["seed"]).shuffle(groups)
        nv, nt = max(1, round(len(groups)*d["val_fraction"])), max(1, round(len(groups)*d["test_fraction"]))
        if nv+nt >= len(groups): raise ValueError("Not enough groups for three splits")
        mapping = {g: "val" if i < nv else "test" if i < nv+nt else "train" for i, g in enumerate(groups)}
    return {name: [r for r in samples if mapping[r["group_id"]] == name] for name in ("train", "val", "test")}, mapping


def sampling_weights(rows, cfg):
    d = cfg["data"]
    bins = np.digitize([r["steering"][0] for r in rows], d["steering_bins"])
    counts = Counter(bins)
    weights = np.array([(len(rows)/(len(counts)*counts[b]))**d["balance_power"] for b in bins])
    weights *= [d["recovery_weight"] if r["recovery"] else 1 for r in rows]
    return np.minimum(weights, d["max_sample_weight"])


def read_prepared(root, cfg):
    """Fail closed on legacy/mixed datasets, malformed labels or split leakage."""
    require_baseline_policy(cfg)
    root = Path(root)
    manifest = json.loads((root/"dataset.json").read_text(encoding="utf-8"))
    if manifest.get("allowed_sources") != ["baseline"]: raise ValueError("Dataset is not certified baseline-only; prepare it again")
    if manifest["contract_hash"] != contract_hash(cfg): raise ValueError("Dataset/model preprocessing contract mismatch")
    sessions = {s["session"]:s for s in manifest["sessions"]}
    groups_seen, sequences_seen, paths_seen, pixels_seen = set(), set(), set(), set()
    result = {}
    h = len(cfg["model"]["horizons_s"])
    for name in ("train", "val", "test"):
        path=root/f"{name}.jsonl"
        if sha256(path) != manifest["split_hashes"][name]: raise ValueError("Dataset was changed after preparation")
        rows=list(read_jsonl(path))
        if not rows or len(rows) != manifest["counts"][name]: raise ValueError(f"Empty or inconsistent {name} split")
        groups, sequences, paths, pixels = set(), set(), set(), set()
        for r in rows:
            if r.get("source") != "baseline": raise ValueError("Non-baseline example in prepared dataset")
            meta=sessions[r["session_id"]]
            if meta["source"] != "baseline" or meta["group"] != r["group_id"] or manifest["groups"][r["group_id"]] != name:
                raise ValueError("Inconsistent session/group provenance")
            if len(r["steering"]) != h or len(r["mask"]) != h or r["mask"][0] != 1:
                raise ValueError("Invalid steering horizons/masks")
            if not all(math.isfinite(v) and -1 <= v <= 1 for v in r["steering"]) or not all(v in (0,1) for v in r["mask"]):
                raise ValueError("Invalid steering labels/masks")
            if not math.isfinite(r["speed"]) or not 0 <= r["speed"] <= cfg["model"]["speed_scale"]:
                raise ValueError("Invalid prepared speed")
            if len(r["images"]) != cfg["model"]["frames"]: raise ValueError("Invalid temporal image count")
            if cfg["model"].get("throttle_output") and not (math.isfinite(r.get("pedal", float("nan"))) and -1 <= r["pedal"] <= 1):
                raise ValueError("Missing or invalid pedal label (prepare the dataset again)")
            groups.add(r["group_id"]); sequences.add(r["sequence_id"].split(":")[0]); paths.update(r["images"])
            pixels.update(r["pixel_sha256"])
        if groups & groups_seen or sequences & sequences_seen or paths & paths_seen or pixels & pixels_seen:
            raise ValueError("Train/validation/test leakage")
        if not np.array([r["mask"] for r in rows]).sum(axis=0).all(): raise ValueError("An output horizon has no supervision")
        groups_seen.update(groups); sequences_seen.update(sequences); paths_seen.update(paths); pixels_seen.update(pixels)
        result[name]=rows
    return manifest,result


def prepare(root, output, cfg):
    output = Path(output)
    if output.exists(): raise FileExistsError(f"Choose a new preparation directory: {output}")
    rows, rejected, sessions = load_sessions(root, cfg)
    splits, groups = split_groups(rows, cfg)
    output.mkdir(parents=True, exist_ok=False)
    dump_json(output/"rejected.json", rejected)
    hashes = {}
    for name, subset in splits.items():
        if not subset: raise ValueError(f"Empty {name} split")
        p = output/f"{name}.jsonl"
        p.write_text("".join(json.dumps(r, allow_nan=False)+"\n" for r in subset), encoding="utf-8")
        hashes[name] = sha256(p)
    report = {"contract_hash": contract_hash(cfg), "allowed_sources": ["baseline"], "split_hashes": hashes, "groups": groups, "counts": {k: len(v) for k,v in splits.items()}, "sessions": sessions, "rejected": len(rejected), "config": cfg}
    dump_json(output/"dataset.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description="Prepare labeled sessions; never connects or trains")
    p.add_argument("--sessions", default="data/sessions")
    p.add_argument("--out", default="data/prepared/v1")
    p.add_argument("--config", default="configs/race.json")
    a = p.parse_args()
    print(json.dumps(prepare(a.sessions, a.out, load_config(a.config)), indent=2))


if __name__ == "__main__": main()
