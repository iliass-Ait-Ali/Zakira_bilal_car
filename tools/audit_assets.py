"""Read-only inspection of source images, with an optional JSON audit report."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from PIL import Image


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--images",default="src/camera")
    p.add_argument("--out",default="docs/assets_audit.json")
    a=p.parse_args()
    counts, hashes, corrupt=Counter(), Counter(), []
    files=sorted(Path(a.images).glob("*"))
    for path in files:
        if not path.is_file(): continue
        try:
            with Image.open(path) as im:
                im.load(); counts[f"{im.width}x{im.height} {im.mode}"]+=1
            hashes[hashlib.sha256(path.read_bytes()).hexdigest()]+=1
        except (OSError,ValueError): corrupt.append(str(path))
    dll=Path("DonkeySimWin/DonkeySimWin/donkey_sim_Data/Managed/Assembly-CSharp.dll")
    report={"image_files":sum(counts.values()),"resolutions":dict(counts),"corrupt":corrupt,"duplicate_file_count":sum(n-1 for n in hashes.values()),"label_sidecars":[str(p) for p in files if p.suffix.lower() in (".json",".jsonl",".csv",".txt")],"assembly_sha256":hashlib.sha256(dll.read_bytes()).hexdigest() if dll.exists() else None}
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    Path(a.out).write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
