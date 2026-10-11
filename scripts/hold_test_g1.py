"""Find a G1 arm pose that holds a regulation basketball between the palms under gravity.

Each env squeezes with a different inward shoulder-roll offset and finger curl. After 1.5 s we report how far the
ball drifted from where it started (small drift = stable hold).
"""

import argparse
import itertools
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from isaaclab_assets.robots.unitree import G1_INSPIRE_FTP_CFG  # noqa: E402

BALL_R = 0.1193  # NBA size 7: 29.5 in circumference
robot_cfg = G1_INSPIRE_FTP_CFG.copy()
robot_cfg.prim_path = "{ENV_REGEX_NS}/Robot"
robot_cfg.spawn.rigid_props.disable_gravity = False
robot_cfg.actuators["arms"] = ImplicitActuatorCfg(
    joint_names_expr=[".*_shoulder_.*", ".*_elbow_joint", ".*_wrist_roll_joint"],
    effort_limit_sim=25.0, velocity_limit_sim=37.0, stiffness=400.0, damping=20.0,
)
robot_cfg.actuators["wrists"] = ImplicitActuatorCfg(
    joint_names_expr=[".*_wrist_pitch_joint", ".*_wrist_yaw_joint"],
    effort_limit_sim=5.0, velocity_limit_sim=22.0, stiffness=100.0, damping=5.0,
)


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot = robot_cfg
    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.30, 0.0, 1.10)),
        spawn=sim_utils.SphereCfg(
            radius=BALL_R,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.62),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8,
                                                            restitution=0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.35, 0.05)),
        ),
    )
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))


ROLL = [0.0, 0.1, 0.2, 0.3, 0.45]
CURL = [0.0, 0.4, 0.8]
CASES = list(itertools.product(ROLL, CURL))

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005))
scene = InteractiveScene(SceneCfg(num_envs=len(CASES), env_spacing=2.0))
sim.reset()
robot, ball = scene["robot"], scene["ball"]
jn = robot.joint_names
palm = [robot.body_names.index(n) for n in ("L_middle_proximal", "R_middle_proximal",
                                              "left_wrist_yaw_link", "right_wrist_yaw_link")]
scene.update(0.0)
p = robot.data.body_pos_w[0, palm] - scene.env_origins[0]
print("[palms] L_mid, R_mid, L_wrist, R_wrist:", [[round(x, 3) for x in r] for r in p.tolist()])

target = robot.data.default_joint_pos.clone()
fingers = [i for i, n in enumerate(jn) if "proximal" in n and "thumb_proximal_pitch" not in n]
for e, (roll, curl) in enumerate(CASES):
    target[e, jn.index("left_shoulder_roll_joint")] = -roll   # inward for the left arm
    target[e, jn.index("right_shoulder_roll_joint")] = roll   # inward for the right arm
    target[e, fingers] = curl

start = ball.data.root_pos_w.clone()
for k in range(300):  # 1.5 s
    robot.set_joint_position_target(target)
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())
drift = torch.norm(ball.data.root_pos_w - start, dim=1)
z = ball.data.root_pos_w[:, 2] - scene.env_origins[:, 2]
print("\n[hold] shoulder_roll_in  finger_curl  ->  ball drift after 1.5 s, ball height")
for e, (roll, curl) in enumerate(CASES):
    tag = "HELD" if drift[e] < 0.08 else "dropped"
    print(f"  roll={roll:.2f}  curl={curl:.1f}  drift={drift[e]:.3f} m  z={z[e]:.2f}  {tag}")
os._exit(0)
