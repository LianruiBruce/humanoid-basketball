"""Roll out a shooting policy and print, per step, where the ball is relative to the palms and why episodes end."""

import argparse
import glob
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Basketball-G1-ShootTrack-Play-v0")
parser.add_argument("--checkpoint", default=None)
parser.add_argument("--steps", type=int, default=25)
parser.add_argument("--zero_action", action="store_true", help="hold the default pose instead of the policy")
parser.add_argument("--ghost_ball", action="store_true", help="ball without collisions (isolates the policy)")
parser.add_argument("--flat_hands", action="store_true", help="fingers straight instead of curled around the ball")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "source"))
import basketball  # noqa: E402, F401
from basketball import tracking_mdp as tm_mdp  # noqa: E402

from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
agent = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
if args.ghost_ball:
    cfg.scene.ball.spawn.collision_props.collision_enabled = False
if args.flat_hands:
    jp = {k: v for k, v in cfg.scene.robot.init_state.joint_pos.items() if "hand" not in k}
    cfg.scene.robot = cfg.scene.robot.replace(init_state=cfg.scene.robot.init_state.replace(joint_pos=jp))
env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg))
base = env.unwrapped
ck = args.checkpoint
if ck is None:
    run = sorted(d for d in glob.glob(os.path.join(ROOT, "logs", "rsl_rl", agent.experiment_name, "*")))[-1]
    ck = max(glob.glob(os.path.join(run, "model_*.pt")), key=lambda p: int(p.split("_")[-1][:-3]))
print(f"[dbg] checkpoint {ck}")
runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=base.device)
runner.load(ck)
policy = runner.get_inference_policy(device=base.device)
robot, ball, tm = base.scene["robot"], base.scene["ball"], base.termination_manager
L, R = robot.body_names.index("left_hand_palm_link"), robot.body_names.index("right_hand_palm_link")
cmd = base.command_manager.get_term("motion")
torch.inference_mode().__enter__()
obs, _ = env.reset()
def palms(e):
    g = tm_mdp.grasp_points(robot.data.body_pos_w[e, [L, R]], robot.data.body_quat_w[e, [L, R]])
    return g[0], g[1]


for k in range(args.steps):
    act = torch.zeros(base.num_envs, base.action_manager.total_action_dim, device=base.device) if args.zero_action \
        else policy(obs)
    obs, _, dones, _ = env.step(act)
    pl, pr = palms(0)
    rel = ball.data.root_pos_w[0] - (pl + pr) / 2
    ql, qr = palms(base.num_envs - 1)
    rel_last = torch.norm(ball.data.root_pos_w[-1] - (ql + qr) / 2).item()
    fired = [n for n in tm.active_terms if tm.get_term(n)[0]]
    print(f"[dbg] step {k:2d} frame {cmd.time_steps[0].item():3d}  palm gap {torch.norm(pl - pr):.3f}  "
          f"ball-palm {[round(x, 3) for x in rel.tolist()]}  |d| {torch.norm(rel):.3f}  "
          f"ball v {[round(x, 2) for x in ball.data.root_lin_vel_w[0].tolist()]}  "
          f"needed {[round(x, 2) for x in tm_mdp.required_release_velocity(base)[0].tolist()]}  "
          f"ball-rim {[round(x, 2) for x in (ball.data.root_pos_w[0] - base.scene['hoop'].data.root_pos_w[0]).tolist()]}"
          f"  {fired}")
os._exit(0)
