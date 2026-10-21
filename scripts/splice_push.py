"""Add a push found by scripts/cem_push.py to a whole-body reference (joint-angle level), ready for prepare_motion.

    python scripts/splice_push.py --ref data/mocap/wholebody/g1_124_04_ball.npz --push data/motions/push_ft.npz \
        --out data/mocap/wholebody/g1_124_04_shot.npz
"""

import argparse

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--ref", required=True)
parser.add_argument("--push", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()

ref = dict(np.load(args.ref))
push = np.load(args.push)
names = [str(n) for n in ref["joint_names"]]
q = ref["joint_pos"].copy()
# replace the push window with the joint angles the robot reached in the replayed push (scripts/cem_push.py
# --replay), blending in over the first BLEND frames and back to the human reference over the last BLEND frames
BLEND = 6
frames = push["reached_frames"]
lo, hi = int(push["frames"][0]), int(push["frames"][-1])
for j, n in enumerate(str(x) for x in push["joint_names"]):
    if n not in names:
        continue
    k = names.index(n)
    for f, val in zip(frames, push["reached_joint_pos"][:, j]):
        if lo <= f <= hi:
            w = min(1.0, (f - lo + 1) / BLEND, (hi - f + 1) / BLEND)
            q[f, k] = (1 - w) * q[f, k] + w * val
ref["joint_pos"] = q
ref["push_frames"] = push["frames"]
ref["push_v_release"] = push["replay_v_release"]
ref["release_frame"] = np.array(int(push["release_frame"]))
np.savez(args.out, **ref)
d = np.abs(q - np.load(args.ref)["joint_pos"]).max(0)
print(f"spliced push into {args.out}: frames {push['frames'][0]}..{push['frames'][-1]}, release {int(push['release_frame'])},"
      f" largest offsets {sorted(zip(d.round(2), names), reverse=True)[:5]}")
