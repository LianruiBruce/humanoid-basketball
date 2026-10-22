"""Search for a push that gives the ball shot speed (cross-entropy method over joint-target offsets).

The trained tracking policy keeps the robot balanced; over the last ``--window`` frames before the release its
joint targets get extra offsets, interpolated through three knots per joint (left/right mirrored). Each env tries
one sample; the cost is the distance between the palm-center velocity at the release and the velocity that drops
the ball through the rim, plus a penalty for falling. The best offsets are saved for splicing into the reference.

    python scripts/cem_push.py --headless --checkpoint <shoot model> --out data/motions/push_ft.npz
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Basketball-G1-ShootTrack-Play-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--iters", type=int, default=15)
parser.add_argument("--window", type=int, default=15)
parser.add_argument("--elite", type=float, default=0.1)
parser.add_argument("--motion", default=None, help="override the reference npz (the push is relative to it)")
parser.add_argument("--reg", type=float, default=0.0, help="cost per rad of mean |offset| (keeps the push close to "
                    "the human motion)")
parser.add_argument("--roll_sigma", type=float, default=0.8, help="initial spread of the shoulder-roll offsets")
parser.add_argument("--leg_sigma", type=float, default=0.3, help="initial spread of the leg / waist offsets")
parser.add_argument("--cost", choices=("velocity", "miss"), default="velocity",
                    help="velocity: distance to the 52 deg release velocity; miss: predicted miss at the rim plane")
parser.add_argument("--w_arm", type=float, default=0.0,
                    help="cost per meter of mean elbow/wrist distance from the human reference during the push")
parser.add_argument("--w_jump", type=float, default=0.0,
                    help="cost per meter the pelvis stays below the reference (keeps the jump) during the push")
parser.add_argument("--init", default=None, help="start the search from the push saved in this file")
parser.add_argument("--init_sigma", type=float, default=0.3, help="spread around --init")
parser.add_argument("--upright_frames", type=int, default=14, help="frames after the release the robot must stay up")
parser.add_argument("--out", required=True)
parser.add_argument("--replay", action="store_true", help="re-run the best push saved in --out and store the joint "
                    "angles the robot actually reached (the reference should hold reachable poses, not PD targets)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
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
cfg.scene.num_envs = args.num_envs
if args.motion:
    cfg.commands.motion.motion_file = args.motion
for t in ("anchor_height", "anchor_tilt", "end_effectors", "made", "missed", "dropped"):
    setattr(cfg.terminations, t, None)  # every env runs the same fixed-length rollout
for r in ("made", "dropped", "fell_height", "fell_tilt", "fell_feet"):
    setattr(cfg.rewards, r, None)  # these read the disabled terminations
cfg.episode_length_s = 100.0
env = RslRlVecEnvWrapper(gym.make(args.task, cfg=cfg))
base = env.unwrapped
runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=base.device)
runner.load(args.checkpoint)
policy = runner.get_inference_policy(device=base.device)
cmd = base.command_manager.get_term("motion")
act = base.action_manager.get_term("joints")
robot = base.scene["robot"]
names = [robot.joint_names[i] for i in act._joint_ids]
dev = base.device

# (joint, mirror sign for the right side) -> parameter; shoulder roll mirrors, the rest are symmetric
GROUPS = [("shoulder_pitch", 1), ("shoulder_roll", -1), ("elbow", 1), ("wrist_pitch", 1), ("hip_pitch", 1),
          ("knee", 1), ("ankle_pitch", 1), ("waist_pitch", 1)]
M = torch.zeros(len(names), len(GROUPS), device=dev)  # joint offsets = M @ group params
for g, (key, mirror) in enumerate(GROUPS):
    for i, n in enumerate(names):
        if key in n:
            M[i, g] = mirror if n.startswith("right") else 1.0
KNOTS = 3
dim = len(GROUPS) * KNOTS
mu = torch.zeros(dim, device=dev)
sigma = torch.tensor([(args.roll_sigma if g == 1 else 0.8) if g < 4 else args.leg_sigma for g in range(len(GROUPS))
                      for _ in range(KNOTS)], device=dev)
if args.init:
    mu = torch.tensor(np.load(args.init)["theta"], dtype=torch.float32, device=dev).flatten()
    sigma = torch.minimum(sigma, torch.full_like(sigma, args.init_sigma))
rel = cmd.release_frame
start = rel - args.window
ARM_BODIES = [i for i, n in enumerate(cmd.cfg.body_names) if "elbow" in n or "wrist" in n]
PELVIS = list(cmd.cfg.body_names).index("pelvis")
print(f"[cem] joints per group: {[(k, int((M[:, g] != 0).sum())) for g, (k, _) in enumerate(GROUPS)]}")


def offsets(theta: torch.Tensor, frame: int) -> torch.Tensor:
    """theta (N, dim) -> joint offsets (N, joints) at ``frame``; linear between knots, decays after the release."""
    th = theta.view(-1, len(GROUPS), KNOTS)
    if frame < start:
        return torch.zeros(theta.shape[0], len(names), device=dev)
    if frame <= rel:
        x = (frame - start) / args.window * (KNOTS - 1)
        k = min(int(x), KNOTS - 2)
        g = th[..., k] + (x - k) * (th[..., k + 1] - th[..., k])
    else:
        g = th[..., -1] * max(0.0, 1.0 - (frame - rel) / 8.0)
    return g @ M.T


def rollout(theta: torch.Tensor):
    obs, _ = env.reset()
    v_rel = torch.zeros(base.num_envs, 3, device=dev)
    v_req = torch.zeros(base.num_envs, 3, device=dev)
    fell = torch.zeros(base.num_envs, dtype=torch.bool, device=dev)
    made = torch.zeros(base.num_envs, dtype=torch.bool, device=dev)
    arm_err = torch.zeros(base.num_envs, device=dev)
    jump_err = torch.zeros(base.num_envs, device=dev)
    prev = torch.full((base.num_envs, 3), float("nan"), device=dev)
    for _ in range(rel + 60 - cmd.time_steps[0].item()):
        frame = cmd.time_steps[0].item()
        a = policy(obs) + offsets(theta, frame) / act._scale
        obs, _, _, _ = env.step(a)
        if frame == rel:
            v_rel = tm_mdp.palm_center_vel(base).clone()
            v_req = tm_mdp.required_release_velocity(base).clone()
        if frame == rel + 3:  # the ball is in free flight: predict where it crosses the rim plane on the way down
            bp = base.scene["ball"].data.root_pos_w - base.scene["hoop"].data.root_pos_w
            bv = base.scene["ball"].data.root_lin_vel_w
            disc = bv[:, 2] ** 2 + 2.0 * 9.81 * bp[:, 2]
            t = (bv[:, 2] + torch.sqrt(disc.clamp(min=0.0))) / 9.81
            miss = torch.norm(bp[:, :2] + bv[:, :2] * t.unsqueeze(1), dim=1)
            miss = torch.where(disc >= 0, miss, torch.norm(bp[:, :2], dim=1) - bp[:, 2])
        if start <= frame <= rel:  # style: stay close to the human motion while pushing
            ref, _ = cmd.aligned_ref_bodies()
            d = torch.norm(ref - cmd.robot_body_pos_w, dim=-1)
            arm_err += d[:, ARM_BODIES].mean(dim=1) / (args.window + 1)
            pz = cmd.robot_body_pos_w[:, PELVIS, 2] - ref[:, PELVIS, 2]
            jump_err += (-pz).clamp(min=0.0) / (args.window + 1)
        if frame <= rel + args.upright_frames:
            h = cmd.robot_anchor_pos_w[:, 2] - cmd.ref_anchor_pos_w[:, 2]
            fell |= (h.abs() > 0.25) | tm_mdp.anchor_tilt_error(base, threshold=0.8)
        # made: the ball center passes down through the rim plane inside the rim
        r = base.scene["ball"].data.root_pos_w - base.scene["hoop"].data.root_pos_w
        made |= (prev[:, 2] > 0) & (r[:, 2] <= 0) & (torch.norm(r[:, :2], dim=1) < 0.2286)
        prev = r.clone()
    err = torch.norm(v_rel - v_req, dim=1) if args.cost == "velocity" else miss
    cost = err + 5.0 * fell.float() - 3.0 * made.float() + args.reg * theta.abs().mean(1)         + args.w_arm * arm_err + args.w_jump * jump_err
    rollout.style = (arm_err, jump_err)
    rollout.made = made
    return cost, err, fell, v_rel, v_req


torch.inference_mode().__enter__()
if args.replay:
    saved = dict(np.load(args.out))
    theta = torch.tensor(saved["theta"], dtype=torch.float32, device=dev).flatten().expand(args.num_envs, -1)
    obs, _ = env.reset()
    frames, reached = [], []
    hoop = base.scene["hoop"].data.root_pos_w[0]
    for _ in range(rel + 60 - cmd.time_steps[0].item()):
        frame = cmd.time_steps[0].item()
        obs, _, _, _ = env.step(policy(obs) + offsets(theta, frame) / act._scale)
        if frame >= rel and (frame - rel) % 4 == 0:
            b = base.scene["ball"].data.root_pos_w[0]
            print(f"[cem] trace frame {frame} ball - rim {[round(x, 2) for x in (b - hoop).tolist()]} "
                  f"ball v {[round(x, 2) for x in base.scene['ball'].data.root_lin_vel_w[0].tolist()]}")
        if frame > rel + 14:
            continue
        frames.append(frame)
        reached.append(robot.data.joint_pos[0, act._joint_ids].cpu().numpy())
        if frame == rel:
            v = tm_mdp.palm_center_vel(base)[0].tolist()
    saved.update(reached_frames=np.array(frames), reached_joint_pos=np.stack(reached), replay_v_release=np.array(v))
    np.savez(args.out, **saved)
    print(f"[cem] replay: release v {[round(x, 2) for x in v]}; reached joint angles for frames {frames[0]}..{frames[-1]}")
    os._exit(0)
best = (float("inf"), None, None)
n_elite = max(int(args.elite * args.num_envs), 4)
for it in range(args.iters):
    theta = mu + sigma * torch.randn(args.num_envs, dim, device=dev)
    theta[0] = mu  # keep the current mean in the population
    cost, err, fell, v_rel, v_req = rollout(theta)
    order = torch.argsort(cost)
    elite = theta[order[:n_elite]]
    mu, sigma = elite.mean(0), elite.std(0).clamp(min=0.05)
    i = order[0].item()
    if cost[i] < best[0]:
        best = (cost[i].item(), theta[i].clone(), v_rel[i].tolist())
    print(f"[cem] iter {it:2d}  best err {err[i]:.2f} m/s (fell {bool(fell[i])})  v {[round(x, 2) for x in v_rel[i].tolist()]}"
          f"  need {[round(x, 2) for x in v_req[i].tolist()]}  mean-sample err {err[0]:.2f} made {bool(rollout.made[0])}"
          f"  population made {rollout.made.float().mean():.2f} fell {fell.float().mean():.2f}"
          f"  best arm err {rollout.style[0][i] * 100:.0f} cm, pelvis below ref {rollout.style[1][i] * 100:.0f} cm",
          flush=True)
theta = best[1].view(len(GROUPS), KNOTS).cpu().numpy()
frames = np.arange(start, rel + 9)
traj = torch.stack([offsets(best[1].unsqueeze(0), int(f))[0] for f in frames]).cpu().numpy()
np.savez(args.out, groups=np.array([k for k, _ in GROUPS]), theta=theta, joint_names=np.array(names),
         frames=frames, offsets=traj, release_frame=rel, cost=best[0], v_release=np.array(best[2]))
print(f"[cem] saved {args.out}: best cost {best[0]:.2f}, release v {[round(x, 2) for x in best[2]]}")
os._exit(0)
