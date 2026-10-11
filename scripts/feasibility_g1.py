"""Can a G1 with real Unitree motor limits throw a 600 g basketball fast enough?

Fixed-base G1, gravity on, arm motors capped at the official limits (25 Nm / 37 rad/s, wrist pitch & yaw
5 Nm / 22 rad/s), 0.3 kg added to each hand to stand in for the ball. Each env tries a different torque-saturated
swing (start pose -> target pose); we report the best palm speed and its elevation angle.
Regulation shots need roughly 7.4 m/s (free throw) and 8.9 m/s (three-pointer) at ~50-55 deg.
"""

import argparse
import itertools
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import math  # noqa: E402

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from isaaclab_assets.robots.unitree import G1_INSPIRE_FTP_CFG  # noqa: E402

ARM_JOINTS = [".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint",
              ".*_wrist_roll_joint"]
WRIST_JOINTS = [".*_wrist_pitch_joint", ".*_wrist_yaw_joint"]

robot_cfg = G1_INSPIRE_FTP_CFG.copy()
robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
robot_cfg.spawn.rigid_props.disable_gravity = False
robot_cfg.actuators["arms"] = ImplicitActuatorCfg(
    joint_names_expr=ARM_JOINTS, effort_limit_sim=25.0, velocity_limit_sim=37.0, stiffness=2000.0, damping=10.0
)
robot_cfg.actuators["wrists"] = ImplicitActuatorCfg(
    joint_names_expr=WRIST_JOINTS, effort_limit_sim=5.0, velocity_limit_sim=22.0, stiffness=500.0, damping=2.0
)


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot = robot_cfg
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))


# swings to try: (shoulder_pitch start, target), (elbow start, target)
SP = [(0.6, -2.6), (0.6, -1.8), (0.0, -2.6), (-0.6, -2.6)]
EL = [(1.6, -1.0), (-0.9, 2.0), (0.0, 0.0)]
SWINGS = list(itertools.product(SP, EL))

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.002))
scene = InteractiveScene(SceneCfg(num_envs=len(SWINGS), env_spacing=2.0))
sim.reset()
robot = scene["robot"]

# add 0.3 kg to each hand (ball stand-in)
masses = robot.root_physx_view.get_masses()
for name in ("left_wrist_yaw_link", "right_wrist_yaw_link"):
    masses[:, robot.body_names.index(name)] += 0.3
robot.root_physx_view.set_masses(masses, torch.arange(scene.num_envs))

jn = robot.joint_names
idx = {n: jn.index(n) for n in jn}
sp_ids = [idx["left_shoulder_pitch_joint"], idx["right_shoulder_pitch_joint"]]
el_ids = [idx["left_elbow_joint"], idx["right_elbow_joint"]]
palm_ids = [robot.body_names.index("L_middle_proximal"), robot.body_names.index("R_middle_proximal")]

start = robot.data.default_joint_pos.clone()
target = start.clone()
for e, ((s0, s1), (e0, e1)) in enumerate(SWINGS):
    start[e, sp_ids], target[e, sp_ids] = s0, s1
    start[e, el_ids], target[e, el_ids] = e0, e1

# settle in the start pose
robot.write_joint_state_to_sim(start, torch.zeros_like(start))
for _ in range(300):
    robot.set_joint_position_target(start)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())

best = torch.zeros(scene.num_envs, device=robot.device)
best_ang = torch.zeros_like(best)
robot.set_joint_position_target(target)
for _ in range(250):  # 0.5 s of full-effort swing
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())
    v = robot.data.body_lin_vel_w[:, palm_ids].mean(dim=1)  # ball ~ midpoint of the two palms
    speed = torch.norm(v, dim=1)
    ang = torch.rad2deg(torch.atan2(v[:, 2], torch.norm(v[:, :2], dim=1)))
    # only count throws that go forward and upward
    useful = (v[:, 0] > 0) & (ang > 20)
    better = useful & (speed > best)
    best = torch.where(better, speed, best)
    best_ang = torch.where(better, ang, best_ang)

print("\n[feasibility] best palm speed per swing (forward & upward only)")
for e, ((s0, s1), (e0, e1)) in enumerate(SWINGS):
    print(f"  shoulder {s0:+.1f}->{s1:+.1f}  elbow {e0:+.1f}->{e1:+.1f}:  {best[e]:.2f} m/s at {best_ang[e]:.0f} deg")
e = int(best.argmax())
print(f"[feasibility] MAX {best[e]:.2f} m/s at {best_ang[e]:.0f} deg  (free throw needs ~7.4, three ~8.9)")
os._exit(0)  # app.close() hangs on this machine
