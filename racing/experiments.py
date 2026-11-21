"""Generate proposed configurations only. Never launches any experiment."""
import argparse
import copy
import json
from pathlib import Path
from .config import load_config, validate_config, dump_json


def main():
    p=argparse.ArgumentParser(description="Write configurations without running training or driving")
    p.add_argument("--config",default="configs/race.json")
    p.add_argument("--space",default="configs/search_space.json")
    p.add_argument("--out",default="configs/proposals")
    a=p.parse_args()
    base=load_config(a.config)
    space=json.loads(Path(a.space).read_text(encoding="utf-8"))
    out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    for stage in space["stages"]:
        for i,changes in enumerate(stage["changes"],1):
            cfg=copy.deepcopy(base)
            for key,value in changes.items():
                section,field=key.split(".")
                if field not in cfg[section]: raise ValueError(f"Unknown parameter {key}")
                cfg[section][field]=value
            validate_config(cfg)
            dump_json(out/f"{stage['name']}_{i}.json",cfg)
    print(f"Configurations written to {out}; no experiment was run.")


if __name__ == "__main__": main()
