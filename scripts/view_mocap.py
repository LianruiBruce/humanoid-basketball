"""Render a CMU clip as stick-figure frames (contact sheet PNG) and print right-wrist height over time.

    python scripts/view_mocap.py data/mocap/cmu_124/124_04.amc --out outputs/mocap_124_04.png
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "source", "basketball"))
from mocap.cmu import load_positions, read_asf  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("amc")
parser.add_argument("--asf", default=None)
parser.add_argument("--out", default=None)
parser.add_argument("--panels", type=int, default=12)
parser.add_argument("--start", type=float, default=0.0)
parser.add_argument("--end", type=float, default=None)
parser.add_argument("--azim", type=float, default=-60)
args = parser.parse_args()
asf = args.asf or os.path.join(os.path.dirname(args.amc), "124.asf")
names, pos = load_positions(asf, args.amc)
bones = read_asf(asf)
idx = {n: i for i, n in enumerate(names)}
T = len(pos)
print(f"{T} frames ({T / 120:.1f} s at 120 Hz)")
rw = pos[:, idx["rwrist"], 2] - pos[:, idx["root"], 2]
lw = pos[:, idx["lwrist"], 2] - pos[:, idx["root"], 2]
for t in range(0, T, 12):
    print(f"  t={t / 120:5.2f}s  R wrist {rw[t]:+.2f} m  L wrist {lw[t]:+.2f} m  (relative to pelvis)")
print(f"peak R wrist at t={np.argmax(rw) / 120:.2f}s, peak L wrist at t={np.argmax(lw) / 120:.2f}s")

edges = [(p, c) for p in bones for c in bones[p].children]
fig = plt.figure(figsize=(3.2 * 6, 4.6 * ((args.panels + 5) // 6)))
center = pos[:, idx["root"]].mean(axis=0)
for k, t in enumerate(np.linspace(int(args.start * 120), int((args.end or (T - 1) / 120) * 120), args.panels).astype(int)):
    ax = fig.add_subplot((args.panels + 5) // 6, 6, k + 1, projection="3d")
    for p, c in edges:
        a, b = pos[t, idx[p]], pos[t, idx[c]]
        color = "tab:red" if c.startswith("r") and c not in ("root",) else "tab:blue" if c.startswith("l") else "k"
        ax.plot(*zip(a, b), color=color, lw=2)
    c = pos[t, idx['root']]
    ax.set_xlim(c[0] - 0.8, c[0] + 0.8)
    ax.set_ylim(c[1] - 0.8, c[1] + 0.8)
    ax.set_zlim(c[2] - 1.0, c[2] + 1.3)
    ax.set_box_aspect((1.6, 1.6, 2.3))
    ax.view_init(elev=8, azim=args.azim)
    ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
    ax.set_title(f"t={t / 120:.2f}s", fontsize=9)
plt.suptitle(os.path.basename(args.amc) + "  (red = right side)")
plt.tight_layout()
out = args.out or os.path.splitext(args.amc)[0] + ".png"
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
plt.savefig(out, dpi=70)
print(f"wrote {out}")
