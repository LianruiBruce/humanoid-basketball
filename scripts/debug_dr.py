"""Ablate domain-randomization components one at a time and report make / miss / drop rates at the free throw."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--drop", nargs="*", default=[], help="event names / 'delay' / 'noise' to disable")
parser.add_argument("--shots", type=int, default=600)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "source"))
import basketball  # noqa: E402, F401
from basketball import g1_mdp  # noqa: E402
from basketball.g1_shoot_env_cfg import ARM_JOINTS, HOLD_OFFSET  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

task = "Basketball-G1-ShootDR-v0"
cfg = load_cfg_from_registry(task, "env_cfg_entry_point")
cfg.scene.num_envs = 512
for name in args.drop:
    if name == "delay":
        cfg.actions.arms = g1_mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=ARM_JOINTS, scale=1.0, use_default_offset=False, offset=HOLD_OFFSET)
    elif name.startswith("maxdelay="):
        cfg.actions.arms.max_delay_substeps = int(name.split("=")[1])
    elif name == "noise":
        cfg.observations.policy.enable_corruption = False
    else:
        setattr(cfg.events, name, None)
env = RslRlVecEnvWrapper(gym.make(task, cfg=cfg))
base = env.unwrapped
agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=base.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=base.device)
ev = base.event_manager.get_term_cfg("reset_hoop").func
ev.frontier, ev.frontier_prob, ev.spread = 4.19, 1.0, 0.0
ev.cfg.params["promote_at"] = 2.0
tm = base.termination_manager
torch.inference_mode().__enter__()
obs, _ = env.reset()
c = {k: 0 for k in ("made_shot", "missed", "dropped", "time_out")}
act = base.action_manager.get_term("arms")
by_delay = {True: [0, 0], False: [0, 0]}  # [made, episodes]
episodes = 0
while episodes < args.shots:
    delayed = act._delayed.clone() if hasattr(act, "_delayed") else torch.zeros(base.num_envs, dtype=torch.bool,
                                                                                 device=base.device)
    obs, _, dones, _ = env.step(policy(obs))
    d = dones.bool()
    made = tm.get_term("made_shot")
    episodes += int(d.sum())
    for flag in (True, False):
        m = d & (delayed == flag)
        by_delay[flag][0] += int((made & m).sum())
        by_delay[flag][1] += int(m.sum())
    for k in c:
        c[k] += int(tm.get_term(k).sum())
print(f"[dr-debug] disabled={args.drop or 'none'}  episodes={episodes}  "
      + "  ".join(f"{k}={v / episodes:.0%}" for k, v in c.items()) + "  (drops also count as misses)")
for flag, (m, n) in by_delay.items():
    print(f"[dr-debug]   delayed={flag}: made {m}/{n} = {m / max(n, 1):.0%}")
os._exit(0)
