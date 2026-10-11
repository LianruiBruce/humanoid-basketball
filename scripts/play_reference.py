"""Kinematically replay a retargeted reference motion on the G1 and record a video (side + front views).

    python scripts/play_reference.py --headless --kit_args="--/app/vulkan=false" --ref data/mocap/g1_124_04.npz
"""

import argparse
import os
import sys

import h5py  # noqa: F401  (see play.py)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--ref", required=True)
parser.add_argument("--loops", type=int, default=2)
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


def cam(name, eye, target):
    return CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}", update_period=0.0, height=480, width=480,
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0),
        offset=CameraCfg.OffsetCfg(pos=eye, rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    ), (eye, target)


side_cam, side_look = cam("SideCam", (0.3, -2.6, 1.1), (0.3, 0.0, 1.1))
front_cam, front_look = cam("FrontCam", (2.6, -0.6, 1.2), (0.0, 0.0, 1.1))


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot = ROBOT_CFG
    floor = AssetBaseCfg(prim_path="/World/floor", spawn=GroundPlaneCfg(color=(0.72, 0.52, 0.32)))
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))
    side = side_cam
    front = front_cam


ref = np.load(args.ref)
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1.0 / float(ref["fps"]), gravity=(0.0, 0.0, 0.0)))
scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
sim.reset()
robot = scene["robot"]
o = scene.env_origins
for c, (eye, look) in ((scene["side"], side_look), (scene["front"], front_look)):
    c.set_world_poses_from_view(torch.tensor([eye], device=o.device) + o, torch.tensor([look], device=o.device) + o)

ids = [robot.joint_names.index(n) for n in ref["joint_names"]]
q_ref = torch.tensor(ref["q"], dtype=torch.float32, device=robot.device)
frames = []
for _ in range(args.loops):
    for t in range(len(q_ref)):
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
imageio.mimsave(out, frames, fps=int(ref["fps"]))
print(f"[replay] wrote {out} ({len(frames)} frames)")
os._exit(0)
