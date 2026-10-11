"""Deterministic hit rate of a G1 shooting policy at fixed distances (curriculum frozen).

    python scripts/eval_g1.py --headless --task Basketball-G1-Shoot-v0 --distances 3.5 4.19 --shots 2000
"""

import argparse
import glob
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Basketball-G1-Shoot-v0")
parser.add_argument("--checkpoint", default=None, help="defaults to the latest run's last checkpoint")
parser.add_argument("--distances", type=float, nargs="+", default=[3.5, 4.19])
parser.add_argument("--shots", type=int, default=2000)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--clean", action="store_true", help="turn off domain randomization, latency and sensor noise")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "source"))
import basketball  # noqa: E402, F401

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
ckpt = args.checkpoint
if ckpt is None:
    run = sorted(glob.glob(os.path.join(ROOT, "logs", "rsl_rl", agent_cfg.experiment_name, "*")))[-1]
    ckpt = max(glob.glob(os.path.join(run, "model_*.pt")), key=lambda p: int(p.split("_")[-1][:-3]))
print(f"[eval] checkpoint: {ckpt}")

results = {}
env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = args.num_envs
if args.clean:
    from basketball import g1_mdp
    from basketball.g1_shoot_env_cfg import ARM_JOINTS, HOLD_OFFSET

    for name in ("ball_mass", "ball_material", "arm_mass", "arm_gains", "arm_joint_friction", "hand_material"):
        if hasattr(env_cfg.events, name):
            setattr(env_cfg.events, name, None)
    env_cfg.actions.arms = g1_mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=ARM_JOINTS, scale=1.0, use_default_offset=False, offset=HOLD_OFFSET)
    env_cfg.observations.policy.enable_corruption = False
    print("[eval] clean env: no randomization, latency or sensor noise")
env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg))
base = env.unwrapped
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=base.device)
runner.load(ckpt)
policy = runner.get_inference_policy(device=base.device)
hoop_event = base.event_manager.get_term_cfg("reset_hoop").func  # reset_hoop_curriculum instance
hoop_event.frontier_prob, hoop_event.spread = 1.0, 0.0  # every episode at exactly the frontier distance
hoop_event.cfg.params["promote_at"] = 2.0  # never promote during evaluation
tm = base.termination_manager

torch.inference_mode().__enter__()  # env buffers become inference tensors once stepped under inference mode
for d in args.distances:
    hoop_event.frontier = d
    obs, _ = env.reset()
    made = missed = timeouts = 0
    while made + missed + timeouts < args.shots:
        obs, _, _, _ = env.step(policy(obs))
        made += int(tm.get_term("made_shot").sum())
        missed += int(tm.get_term("missed").sum())
        timeouts += int(tm.get_term("time_out").sum())
    total = made + missed + timeouts
    results[d] = made / total
    print(f"[eval] distance={d:.2f} m  shots={total}  made={made}  hit_rate={made / total:.1%}")

print("[eval] summary: " + "  ".join(f"{d:.2f}m={r:.0%}" for d, r in results.items()))
os._exit(0)
