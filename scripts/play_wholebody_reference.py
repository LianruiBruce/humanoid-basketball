"""Kinematically replay a whole-body G1 reference (root pose + 29 joints) and record side + front views.

    python scripts/play_wholebody_reference.py --headless --kit_args="--/app/vulkan=false" \
        --ref data/mocap/wholebody/g1_124_05.npz
"""

import argparse
import os
import sys

import h5py  # noqa: F401  (see play.py)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--ref", required=True)
parser.add_argument("--loops", type=int, default=1)
parser.add_argument("--out", default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app = AppLauncher(args).app

import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "source"))
from basketball.g1_shoot_env_cfg import ROBOT_CFG  # noqa: E402

# The G1 + Inspire USD carries a fixed root joint and cannot be made floating (PhysX refuses the articulation),
# so for a kinematic replay the fixed base is teleported to the reference root pose every frame.
robot_cfg = ROBOT_CFG.copy()
robot_cfg.spawn.rigid_props.disable_gravity = True


def cam(name, eye):
    return CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}", update_period=0.0, height=540, width=480,
        spawn=sim_utils.PinholeCameraCfg(focal_length=16.0),
        offset=CameraCfg.OffsetCfg(pos=eye, rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    )


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot = robot_cfg
    floor = AssetBaseCfg(prim_path="/World/floor", spawn=GroundPlaneCfg(color=(0.72, 0.52, 0.32)))
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    side = cam("SideCam", (0.2, -3.0, 0.9))
    front = cam("FrontCam", (3.0, -0.8, 1.0))


ref = np.load(args.ref)
fps = float(ref["fps"])
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / fps, gravity=(0.0, 0.0, 0.0)))
scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
sim.reset()
robot = scene["robot"]
dev = robot.device
o = scene.env_origins
look = torch.tensor([[0.2, 0.0, 0.75]], device=dev) + o
scene["side"].set_world_poses_from_view(torch.tensor([[0.2, -3.0, 0.9]], device=dev) + o, look)
scene["front"].set_world_poses_from_view(torch.tensor([[3.0, -0.8, 1.0]], device=dev) + o, look)

ids = [robot.joint_names.index(str(n)) for n in ref["joint_names"]]
q_ref = torch.tensor(ref["joint_pos"], dtype=torch.float32, device=dev)
root_pos = torch.tensor(ref["root_pos"], dtype=torch.float32, device=dev)
root_quat = torch.tensor(ref["root_quat"], dtype=torch.float32, device=dev)
frames = []
for _ in range(args.loops):
    for t in range(len(q_ref)):
        pose = torch.cat([root_pos[t] + o[0], root_quat[t]]).unsqueeze(0)
        robot.write_root_pose_to_sim(pose)
        robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))
        q = robot.data.default_joint_pos.clone()
        q[0, ids] = q_ref[t]
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        robot.set_joint_position_target(q)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
        a = scene["side"].data.output["rgb"][0, ..., :3].cpu().numpy()
        b = scene["front"].data.output["rgb"][0, ..., :3].cpu().numpy()
        frames.append(np.concatenate([a, b], axis=1))
out = args.out or os.path.splitext(args.ref)[0] + "_replay.mp4"
os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
imageio.mimsave(out, frames, fps=int(fps))
print(f"[replay] wrote {out} ({len(frames)} frames)")
os._exit(0)
