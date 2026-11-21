"""Plot a real training metrics.csv after training has been run by the user."""
import argparse
import csv
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("metrics")
    p.add_argument("--out", default="training_metrics.png")
    a = p.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with Path(a.metrics).open(encoding="utf-8") as f: rows=list(csv.DictReader(f))
    if not rows: raise ValueError("No recorded epochs")
    x=[int(r["epoch"]) for r in rows]
    fig, axes=plt.subplots(1,2,figsize=(10,4))
    for key in ("train_loss","val_loss"): axes[0].plot(x,[float(r[key]) for r in rows],label=key)
    for key in ("val_mae","val_turn_mae"): axes[1].plot(x,[float(r[key]) if r[key] else float("nan") for r in rows],label=key)
    for ax in axes: ax.set_xlabel("Epoch"); ax.legend(); ax.grid(alpha=.2)
    axes[0].set_ylabel("Weighted Huber loss"); axes[1].set_ylabel("Normalized steering MAE")
    fig.tight_layout(); fig.savefig(a.out,dpi=160); plt.close(fig)


if __name__ == "__main__": main()
