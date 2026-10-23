"""Plot the Agron domain-pretraining run (train loss + held-out source losses).

Usage:
  python plot_domain_training.py --run-dir /content/agron-domain-run
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    with open(os.path.join(args.run_dir, "report.json")) as f:
        report = json.load(f)
    with open(os.path.join(args.run_dir, "eval_history.json")) as f:
        history = json.load(f)

    out = args.output or os.path.join(args.run_dir, "domain_pretraining_curve.png")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=140)

    train = report.get("train_loss_history", [])
    if train:
        axes[0].plot(
            [e["step"] for e in train],
            [e["loss"] for e in train],
            color="#2563eb",
            lw=1.2,
            label="train loss",
        )
    axes[0].set_xlabel("optimizer step")
    axes[0].set_ylabel("cross-entropy loss")
    axes[0].set_title("Agron domain pretraining - train loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend(frameon=False)

    steps = [0] + [e["step"] for e in history]
    colors = {"code": "#16a34a", "cyber": "#dc2626", "replay": "#7c3aed"}
    for source in ("code", "cyber", "replay"):
        before = report.get("eval_before", {}).get(source, {}).get("loss")
        if before is None:
            continue
        ys = [before] + [e[source]["loss"] for e in history if source in e]
        xs = [0] + [e["step"] for e in history if source in e]
        axes[1].plot(xs, ys, marker="o", ms=3, lw=1.5, color=colors[source], label=source)
    axes[1].set_xlabel("optimizer step")
    axes[1].set_ylabel("held-out loss")
    axes[1].set_title("Held-out loss per source")
    axes[1].grid(alpha=0.3)
    axes[1].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(out)
    print("saved", out)


if __name__ == "__main__":
    main()
