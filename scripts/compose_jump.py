"""Add a human vertical jump to the free throw: upper body (waist, arms, pelvis orientation) from the free-throw clip,
legs and pelvis height from the jump-shot clip, time-shifted so the jump's apex lands at the free-throw release.

    # 1. compose, 2. forward kinematics, 3. put the stance feet back on the floor, 4. FK again
    python scripts/compose_jump.py --out data/mocap/wholebody/g1_124_04_jump.npz
    python scripts/prepare_motion.py --headless --ref data/mocap/wholebody/g1_124_04_jump.npz --out data/motions/g1_124_04_jump.npz
    python scripts/compose_jump.py --out data/mocap/wholebody/g1_124_04_jump.npz --fk data/motions/g1_124_04_jump.npz
    python scripts/prepare_motion.py --headless --ref data/mocap/wholebody/g1_124_04_jump.npz --out data/motions/g1_124_04_jump.npz
"""

import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--upper", default="data/mocap/wholebody/g1_124_04_ball.npz", help="free throw (upper body)")
parser.add_argument("--lower", default="data/mocap/wholebody/g1_124_05_ball.npz", help="jump shot (legs, pelvis z)")
parser.add_argument("--shift", type=int, default=7, help="lower-clip frame = upper-clip frame - shift")
parser.add_argument("--window", type=int, nargs=2, default=(22, 75), help="upper-clip frames that take the jump")
parser.add_argument("--blend", type=int, default=6)
parser.add_argument("--airborne", type=int, nargs=2, default=(36, 63),
                    help="upper-clip frames with the feet off the floor (no ground correction there)")
parser.add_argument("--fk", default=None, help="prepared motion of the composed clip: fix the stance foot height")
parser.add_argument("--sole", type=float, default=0.036, help="ankle-roll link height with the foot flat")
parser.add_argument("--out", required=True)
args = parser.parse_args()

if args.fk:
    # second pass: lift / lower the pelvis so the lower ankle sits at sole height on stance frames; interpolate the
    # correction across the flight so the jump height is kept
    ref = dict(np.load(args.out))
    fk = np.load(args.fk)
    names = [str(n) for n in fk["body_names"]]
    feet = [names.index("left_ankle_roll_link"), names.index("right_ankle_roll_link")]
    low = fk["body_pos_w"][:, feet, 2].min(axis=1)
    corr = args.sole - low
    t = np.arange(len(corr))
    a, b = args.airborne
    stance = (t < a) | (t > b)
    corr = np.interp(t, t[stance], corr[stance])
    ref["root_pos"] = ref["root_pos"].copy()
    ref["root_pos"][:, 2] += corr
    np.savez(args.out, **ref)
    print(f"[compose] ground fix: correction {corr.min() * 100:+.1f}..{corr.max() * 100:+.1f} cm, "
          f"flight clearance {(low[a:b + 1] + corr[a:b + 1] - args.sole).max() * 100:.1f} cm")
    raise SystemExit

up = dict(np.load(args.upper))
lo = np.load(args.lower)
names = [str(n) for n in up["joint_names"]]
assert names == [str(n) for n in lo["joint_names"]]
legs = [i for i, n in enumerate(names) if any(k in n for k in ("hip", "knee", "ankle"))]
T = len(up["joint_pos"])
s0, s1 = args.window
w = np.zeros(T)
for t in range(T):
    if s0 <= t <= s1:
        w[t] = min(1.0, (t - s0 + 1) / args.blend, (s1 - t + 1) / args.blend)
src = np.clip(np.arange(T) - args.shift, 0, len(lo["joint_pos"]) - 1)

q = up["joint_pos"].copy()
q[:, legs] = (1 - w[:, None]) * q[:, legs] + w[:, None] * lo["joint_pos"][src][:, legs]
root = up["root_pos"].copy()
# the jump's pelvis height, offset so the two clips agree where the window opens
dz = up["root_pos"][s0, 2] - lo["root_pos"][src[s0], 2]
root[:, 2] = (1 - w) * up["root_pos"][:, 2] + w * (lo["root_pos"][src, 2] + dz)
up["joint_pos"], up["root_pos"] = q, root
up["composed_from"] = np.array([args.upper, args.lower])
up["jump_shift"] = np.array(args.shift)
np.savez(args.out, **up)
print(f"[compose] {args.out}: legs {len(legs)} joints from {args.lower} (shift {args.shift}), window {s0}..{s1}, "
      f"pelvis z {root[:, 2].min():.2f}..{root[:, 2].max():.2f} m")
