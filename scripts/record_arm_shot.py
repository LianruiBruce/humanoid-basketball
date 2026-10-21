"""Record the arm joint trajectory of made shots from the arm-only shooting policy (fixed pelvis, 85-90% makes).

The whole-body reference from mocap has no wrist / finger flick, so its push is far too slow for a shot; this
trajectory is spliced into it over the push phase (scripts/splice_arm_shot.py).

    python scripts/record_arm_shot.py --headless --checkpoint checkpoints/g1_amp.pt --distance 4.19 \
        --out data/motions/arm_shot_ft.npz
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Basketball-G1-ShootAMP-Play-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--distance", type=float, default=4.19)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--pre", type=int, default=25, help="frames kept before the release")
parser.add_argument("--post", type=int, default=10, help="frames kept after the release")
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "source"))
import basketball  # noqa: E402, F401
from basketball import g1_mdp  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = args.num_envs
env_cfg.events.reset_hoop.params["lateral"] = 0.0
env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg))
base = env.unwrapped
hoop_event = base.event_manager.get_term_cfg("reset_hoop").func
hoop_event.frontier, hoop_event.frontier_prob, hoop_event.spread = args.distance, 1.0, 0.0
hoop_event.cfg.params["promote_at"] = 2.0

runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=base.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=base.device)
robot = base.scene["robot"]
arm_names = [n for n in robot.joint_names if any(k in n for k in ("shoulder", "elbow", "wrist"))]
arm_ids = [robot.joint_names.index(n) for n in arm_names]
tm = base.termination_manager

torch.inference_mode().__enter__()
obs, _ = env.reset()
hist = [[] for _ in range(base.num_envs)]
release = [-1] * base.num_envs
shots = []
n_made, rel_at = 0, []
for _ in range(400):
    q = robot.data.joint_pos[:, arm_ids].cpu().numpy()
    rel = g1_mdp.ball_released(base).cpu().numpy()
    for e in range(base.num_envs):
        hist[e].append(q[e])
        if rel[e] and release[e] < 0:
            release[e] = len(hist[e]) - 1
    obs, _, dones, _ = env.step(policy(obs))
    made = tm.get_term("made_shot").cpu().numpy()
    for e in np.nonzero(dones.cpu().numpy())[0]:
        r = release[e]
        n_made += int(made[e])
        if made[e]:
            rel_at.append(r)
        if made[e] and r >= args.pre and len(hist[e]) > r + args.post:
            shots.append(np.stack(hist[e][r - args.pre:r + args.post + 1]))
        hist[e], release[e] = [], -1
print(f"[arm] {n_made} made, release steps {sorted(rel_at)[:20]}; {len(shots)} kept")
traj = np.mean(np.stack(shots), axis=0) if shots else None
if traj is not None:
    np.savez(args.out, joint_names=np.array(arm_names), joint_pos=traj, release_index=args.pre, fps=50.0,
             shots=len(shots), example=shots[0])
    print(f"[arm] saved {args.out}: {traj.shape}, joints {arm_names}")
os._exit(0)
