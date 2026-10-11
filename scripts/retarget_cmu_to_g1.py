"""Retarget the arm motion of a CMU mocap clip onto the G1 (fixed torso) and save joint trajectories.

For every frame we express the human upper-arm, forearm and hand directions in the human torso frame, rebuild
elbow / wrist / knuckle targets with the G1's own segment lengths, and solve damped-least-squares IK for both
7-DoF arms. Every frame lives in its own env, so all frames are solved in parallel.

    python scripts/retarget_cmu_to_g1.py --headless --amc data/mocap/cmu_124/124_04.amc --start 3.7 --end 5.2 \
        --out data/mocap/g1_124_04.npz
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--amc", required=True)
parser.add_argument("--asf", default=None)
parser.add_argument("--start", type=float, required=True, help="seconds")
parser.add_argument("--end", type=float, required=True, help="seconds")
parser.add_argument("--fps", type=float, default=50.0, help="output rate (policy runs at 50 Hz)")
parser.add_argument("--iters", type=int, default=80)
parser.add_argument("--out", required=True)
parser.add_argument("--one_hand", action="store_true",
                    help="right-handed NBA form: freeze the left (guide) arm once the ball reaches the set point")
parser.add_argument("--set_fraction", type=float, default=0.8,
                    help="set point = first frame the right wrist reaches this fraction of its peak height")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "source"))
from basketball.g1_shoot_env_cfg import ROBOT_CFG  # noqa: E402
from basketball.mocap.cmu import load_positions  # noqa: E402

SIDES = {"left": ("l", "L"), "right": ("r", "R")}
ARM = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]


def human_targets(names, pos):
    """Unit directions of upper arm / forearm / hand per side, in the human torso frame (x fwd, y left, z up)."""
    i = {n: k for k, n in enumerate(names)}
    up = pos[:, i["thorax"]] - pos[:, i["root"]]
    up /= np.linalg.norm(up, axis=1, keepdims=True)
    left = pos[:, i["lclavicle"]] - pos[:, i["rclavicle"]]
    fwd = np.cross(left, up)
    fwd /= np.linalg.norm(fwd, axis=1, keepdims=True)
    left = np.cross(up, fwd)
    R = np.stack([fwd, left, up], axis=1)  # rows = torso axes -> world-to-torso rotation
    out = {}
    for side, (h, _) in SIDES.items():
        sh, el, wr, hd = (pos[:, i[f"{h}clavicle"]], pos[:, i[f"{h}humerus"]], pos[:, i[f"{h}radius"]],
                          pos[:, i[f"{h}hand"]])
        dirs = []
        for a, b in ((sh, el), (el, wr), (wr, hd)):
            d = np.einsum("tij,tj->ti", R, b - a)
            dirs.append(d / np.linalg.norm(d, axis=1, keepdims=True))
        out[side] = dirs
    return out


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot = ROBOT_CFG.replace(init_state=ROBOT_CFG.init_state.replace(pos=(0.0, 0.0, 0.793)))


def main():
    asf = args.asf or os.path.join(os.path.dirname(args.amc), "124.asf")
    names, pos = load_positions(asf, args.amc)
    t_src = np.arange(len(pos)) / 120.0
    t_out = np.arange(args.start, args.end, 1.0 / args.fps)
    pos = np.stack([np.stack([np.interp(t_out, t_src, pos[:, j, k]) for k in range(3)], -1)
                    for j in range(pos.shape[1])], 1)
    dirs = human_targets(names, pos)
    T = len(t_out)
    if args.one_hand:
        # guide hand: stays at the side of the ball from the set point on instead of pushing with the shooting hand
        i = {n: k for k, n in enumerate(names)}
        rw = pos[:, i["rwrist"], 2] - pos[:, i["root"], 2]
        rw_rel = (rw - rw.min()) / max(rw.max() - rw.min(), 1e-6)
        t_set = int(np.argmax(rw_rel >= args.set_fraction))
        for k in range(3):
            dirs["left"][k][t_set:] = dirs["left"][k][t_set]
        print(f"[retarget] one-hand form: left arm frozen from t={t_out[t_set]:.2f}s (frame {t_set}/{T})")
    print(f"[retarget] {T} frames at {args.fps:.0f} Hz from {os.path.basename(args.amc)} [{args.start}, {args.end}] s")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, gravity=(0.0, 0.0, 0.0)))
    scene = InteractiveScene(SceneCfg(num_envs=T, env_spacing=2.0))
    sim.reset()
    robot = scene["robot"]
    dev = robot.device
    bn, jn = robot.body_names, robot.joint_names

    # G1 segment lengths and shoulder anchor (torso frame == env frame: fixed pelvis, waist at 0)
    scene.update(0.0)
    origin = scene.env_origins
    bpos = robot.data.body_pos_w[0] - origin[0]
    chains, anchors, lengths = {}, {}, {}
    for side, (h, H) in SIDES.items():
        bodies = [f"{side}_shoulder_roll_link", f"{side}_elbow_link", f"{side}_wrist_yaw_link", f"{H}_middle_proximal"]
        ids = [bn.index(b) for b in bodies]
        chains[side] = ids
        anchors[side] = bpos[ids[0]]
        lengths[side] = [torch.norm(bpos[ids[k + 1]] - bpos[ids[k]]).item() for k in range(3)]
        print(f"[retarget] G1 {side} segments (m): upper {lengths[side][0]:.3f} fore {lengths[side][1]:.3f} "
              f"hand {lengths[side][2]:.3f}")

    # targets per env: elbow, wrist, knuckles
    targets = {}
    for side in SIDES:
        p = anchors[side].expand(T, 3).clone()
        tg = []
        for k in range(3):
            p = p + torch.tensor(dirs[side][k], dtype=torch.float32, device=dev) * lengths[side][k]
            tg.append(p.clone())
        targets[side] = torch.stack(tg, 1) + origin[:, None, :]  # (T, 3, 3) world

    joint_ids = {s: [jn.index(f"{s}_{a}_joint") for a in ARM] for s in SIDES}
    q = robot.data.default_joint_pos.clone()
    lo, hi = robot.data.soft_joint_pos_limits[..., 0], robot.data.soft_joint_pos_limits[..., 1]
    weights = torch.tensor([0.5, 1.0, 1.0], device=dev)
    lam = 0.05
    for it in range(args.iters):
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        sim.forward()
        scene.update(0.0)
        jac = robot.root_physx_view.get_jacobians()  # (T, bodies-1, 6, dofs) for a fixed base
        err_total = 0.0
        for side in SIDES:
            cur = robot.data.body_pos_w[:, chains[side][1:]]  # (T, 3, 3)
            e = (targets[side] - cur) * weights[None, :, None]
            J = torch.stack([jac[:, b - 1, :3][:, :, joint_ids[side]] for b in chains[side][1:]], 1)
            J = (J * weights[None, :, None, None]).reshape(T, 9, 7)
            e = e.reshape(T, 9, 1)
            JT = J.transpose(1, 2)
            dq = JT @ torch.linalg.solve(J @ JT + lam**2 * torch.eye(9, device=dev), e)
            q[:, joint_ids[side]] += dq.squeeze(-1).clamp(-0.2, 0.2)
            err_total += torch.norm(targets[side] - cur, dim=-1).mean().item()
        q = torch.max(torch.min(q, hi), lo)
        if it % 20 == 0 or it == args.iters - 1:
            print(f"[retarget] iter {it:3d}  mean target error {err_total / 2 * 100:.1f} cm")

    # save arm joint trajectory + key body positions (torso frame) for the motion prior
    robot.write_joint_state_to_sim(q, torch.zeros_like(q))
    sim.forward()
    scene.update(0.0)
    arm_ids = joint_ids["left"] + joint_ids["right"]
    key_ids = chains["left"][1:] + chains["right"][1:]
    key = (robot.data.body_pos_w[:, key_ids] - origin[:, None, :]).cpu().numpy()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, joint_names=np.array([jn[i] for i in arm_ids]), q=q[:, arm_ids].cpu().numpy(),
             key_body_names=np.array([bn[i] for i in key_ids]), key_pos=key, fps=args.fps,
             source=os.path.basename(args.amc), window=np.array([args.start, args.end]))
    print(f"[retarget] saved {args.out}: q {tuple(q[:, arm_ids].shape)}")


main()
os._exit(0)
