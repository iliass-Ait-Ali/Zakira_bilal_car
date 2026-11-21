"""Compare reviewed real-run reports; no scoring formula is invented."""
import argparse
import json
from pathlib import Path
from .config import dump_json


def compare(reports):
    usable=[r for r in reports if r.get("success_rate") is not None and r.get("lap_mean_s") is not None]
    frontier=[]
    for i,r in enumerate(usable):
        dominated=any(j!=i and s["success_rate"]>=r["success_rate"] and s["lap_mean_s"]<=r["lap_mean_s"] and (s["success_rate"]>r["success_rate"] or s["lap_mean_s"]<r["lap_mean_s"]) for j,s in enumerate(usable))
        if not dominated: frontier.append(i)
    return {"reports":reports,"reviewed_reports":usable,"pareto_indices_in_reviewed_reports":frontier,"note":"Descriptive speed/success comparison only. Equal track, conditions, attempts and independent runs must be checked using run.json and the evaluation protocol; small samples do not establish superiority."}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("reports",nargs="+")
    p.add_argument("--out",required=True)
    a=p.parse_args()
    dump_json(a.out,compare([json.loads(Path(p).read_text(encoding="utf-8")) for p in a.reports]))


if __name__ == "__main__": main()
