"""Print the G1 + Inspire hand structure: joints, limits, bodies, and key body heights at the default pose."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from isaaclab_assets.robots.unitree import G1_INSPIRE_FTP_CFG  # noqa: E402


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot = G1_INSPIRE_FTP_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))


sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))
scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
sim.reset()
scene.update(0.0)

robot = scene["robot"]
d = robot.data
print(f"\n[joints] {robot.num_joints}")
lim = d.joint_pos_limits[0]
for i, n in enumerate(robot.joint_names):
    print(f"  {i:2d} {n:40s} [{lim[i, 0]:+.2f}, {lim[i, 1]:+.2f}]  vel_lim={d.joint_vel_limits[0, i]:.1f}"
          f"  effort_lim={d.joint_effort_limits[0, i]:.1f}")

print(f"\n[bodies] {robot.num_bodies}")
pos = d.body_pos_w[0] - scene.env_origins[0]
mass = robot.root_physx_view.get_masses()[0]
for i, n in enumerate(robot.body_names):
    print(f"  {i:2d} {n:40s} pos={[round(v, 3) for v in pos[i].tolist()]}  mass={mass[i]:.3f}")
print(f"\n[total mass] {mass.sum():.2f} kg")

app.close()
