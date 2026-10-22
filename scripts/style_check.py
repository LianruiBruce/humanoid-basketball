"""How close is a shooting policy to the human reference? Per reference frame: mean body-position error (after
heading alignment, cm), pelvis height and foot clearance of robot vs reference.

    python scripts/style_check.py --headless --task Basketball-G1-JumpShotPush-Play-v0 --checkpoint <ckpt>
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Basketball-G1-JumpShotPush-Play-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "source"))
import basketball  # noqa: E402, F401

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
agent = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
cfg.scene.num_envs = args.num_envs
env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg))
base = env.unwrapped
runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=base.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=base.device)
c = base.command_manager.get_term("motion")
names = list(c.cfg.body_names)
arm = [i for i, n in enumerate(names) if "elbow" in n or "wrist" in n]
feet = [i for i, n in enumerate(names) if "ankle" in n]
pelvis = names.index("pelvis")
o = base.scene.env_origins
torch.inference_mode().__enter__()
obs, _ = env.reset()
rows = {}
for _ in range(c.release_frame + 30):
    f = c.time_steps[0].item()
    ref, _ = c.aligned_ref_bodies()
    rob = c.robot_body_pos_w
    err = torch.norm(ref - rob, dim=-1)  # (envs, bodies)
    rows[f] = (err.mean().item() * 100, err[:, arm].mean().item() * 100,
               (rob[:, pelvis, 2] - o[:, 2]).mean().item(), (ref[:, pelvis, 2] - o[:, 2]).mean().item(),
               (rob[:, feet, 2] - o[:, 2].unsqueeze(1)).min(dim=1).values.mean().item(),
               (ref[:, feet, 2] - o[:, 2].unsqueeze(1)).min(dim=1).values.mean().item())
    obs, _, _, _ = env.step(policy(obs))
R = c.release_frame
print(f"[style] {args.task}: release frame {R}")
print("[style] frame  body err cm  arm err cm   pelvis z robot/ref   lower foot z robot/ref")
for f in sorted(rows):
    if f % 3 == 0 or f == R:
        e, ea, pz, prz, fz, frz = rows[f]
        mark = " <- release" if f == R else ""
        print(f"[style] {f:5d}  {e:10.1f}  {ea:10.1f}   {pz:6.2f} / {prz:5.2f}       {fz:6.3f} / {frz:5.3f}{mark}")
os._exit(0)
