"""Turn a whole-body reference (root pose + 29 joint angles) into a tracking target for the floating-base G1.

Writes every frame onto the robot (one env per frame), runs forward kinematics and stores body positions,
orientations and finite-difference velocities, the format the motion-tracking task reads.

    python scripts/prepare_motion.py --headless --ref data/mocap/wholebody/g1_124_04.npz \
        --out data/motions/g1_124_04.npz
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--ref", required=True)
parser.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "source"))
from basketball.tracking_env_cfg import G1_TRACKING_CFG  # noqa: E402

robot_cfg = G1_TRACKING_CFG.copy()
robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
robot_cfg.spawn.rigid_props.disable_gravity = True


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot = robot_cfg


ref = np.load(args.ref)
T = len(ref["joint_pos"])
fps = float(ref["fps"])
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / fps, gravity=(0.0, 0.0, 0.0)))
scene = InteractiveScene(SceneCfg(num_envs=T, env_spacing=3.0))
sim.reset()
robot = scene["robot"]
dev = robot.device
o = scene.env_origins
print(f"[prep] robot: {robot.num_joints} joints, {robot.num_bodies} bodies, fixed base: {robot.is_fixed_base}")

ref_joints = [str(n) for n in ref["joint_names"]]
missing = [n for n in ref_joints if n not in robot.joint_names]
extra = [n for n in robot.joint_names if n not in ref_joints]
print(f"[prep] joints missing on robot: {missing}  robot joints not in reference: {extra}")
ids = [robot.joint_names.index(n) for n in ref_joints]

tt = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)  # noqa: E731
q = robot.data.default_joint_pos.clone()
q[:, ids] = tt(ref["joint_pos"])
root = torch.cat([tt(ref["root_pos"]) + o, tt(ref["root_quat"])], dim=1)
robot.write_root_pose_to_sim(root)
robot.write_root_velocity_to_sim(torch.zeros(T, 6, device=dev))
robot.write_joint_state_to_sim(q, torch.zeros_like(q))
sim.forward()
scene.update(0.0)

body_pos = (robot.data.body_pos_w - o[:, None, :]).cpu().numpy()
body_quat = robot.data.body_quat_w.cpu().numpy()
joint_pos = q.cpu().numpy()


def central_diff(x):
    v = np.gradient(x, axis=0) * fps
    return v


bq = torch.tensor(body_quat)
dq = quat_mul(bq[2:], quat_conjugate(bq[:-2]))  # rotation over two frames
ang = axis_angle_from_quat(dq).numpy() * fps / 2.0
body_ang_vel = np.concatenate([ang[:1], ang, ang[-1:]], axis=0)

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
np.savez(args.out, fps=fps, joint_names=np.array(robot.joint_names), joint_pos=joint_pos,
         joint_vel=central_diff(joint_pos), body_names=np.array(robot.body_names), body_pos_w=body_pos,
         body_quat_w=body_quat, body_lin_vel_w=central_diff(body_pos), body_ang_vel_w=body_ang_vel,
         source=str(ref["source"]))
sole = body_pos[:, [robot.body_names.index("left_ankle_roll_link"), robot.body_names.index("right_ankle_roll_link")], 2]
print(f"[prep] {T} frames, pelvis z {body_pos[:, 0, 2].min():.2f}-{body_pos[:, 0, 2].max():.2f} m, "
      f"ankle z min {sole.min():.3f} m, max joint speed {np.abs(central_diff(joint_pos)).max():.1f} rad/s")
print(f"[prep] saved {args.out}")
os._exit(0)
