"""Fait conduire le reseau v14 (100 % deep learning) sur un circuit.
Usage : python lancer.py [circuit] [secondes]   (sans argument : un menu s'affiche)"""
import json, socket, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
CIRCUITS = ["roboracingleague_1", "warren", "mini_monaco", "generated_road", "generated_track",
            "waveshare", "warehouse", "circuit_launch", "mountain_track", "sparkfun_avc"]
SIM = ROOT/"DonkeySimWin"/"DonkeySimWin"/"donkey_sim.exe"
CHECKPOINT = ROOT/"runs"/"cnn_all10_v14"/"best.pt"


def choose():
    print("Circuits :")
    for i, c in enumerate(CIRCUITS, 1): print(f"  {i:2d}. {c}")
    while True:
        answer = input("Numero du circuit (1-10) : ").strip()
        if answer.isdigit() and 1 <= int(answer) <= 10: return CIRCUITS[int(answer)-1]
        print("Tape un nombre entre 1 et 10.")


def port_open(port=9091):
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1): return True
    except OSError:
        return False


def main():
    scene = sys.argv[1] if len(sys.argv) > 1 else choose()
    if scene not in CIRCUITS: sys.exit(f"Circuit inconnu : {scene}. Choix : {', '.join(CIRCUITS)}")
    seconds = sys.argv[2] if len(sys.argv) > 2 else "120"
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"  # carte graphique NVIDIA si disponible
    cfg = json.loads((ROOT/"configs"/"tracks"/f"cnn_v14n_{scene}.json").read_text(encoding="utf-8"))
    cfg["runtime"]["inference_device"] = device
    out = ROOT/"runs"/"essais"; out.mkdir(parents=True, exist_ok=True)
    config = out/f"config_{scene}.json"
    config.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"Circuit {scene}, {seconds} s, reseau sur {device}. Ouverture du simulateur...")
    if port_open(): sys.exit("Un simulateur est deja ouvert : ferme-le puis relance.")
    sim = subprocess.Popen([str(SIM)], cwd=str(SIM.parent))
    try:
        for _ in range(90):
            if port_open(): break
            time.sleep(1)
        else:
            sys.exit("Le simulateur ne repond pas (port 9091).")
        time.sleep(3)
        code = subprocess.call([sys.executable, "-m", "racing.drive", "--connect", "--mode", "cnn",
                                "--checkpoint", str(CHECKPOINT), "--config", str(config),
                                "--seconds", str(seconds), "--logdir", str(out)], cwd=str(ROOT))
        print("Termine." if code == 0 else f"Arret avec le code {code} (voir les messages ci-dessus).")
    finally:
        time.sleep(1)
        if sim.poll() is None: sim.kill()


if __name__ == "__main__":
    main()
