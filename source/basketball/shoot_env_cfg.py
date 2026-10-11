"""Franka shoots a ball into a hoop placed at a random position beyond the table.

Each episode is one shot: the ball starts in the gripper, the policy swings the arm (joint position targets)
and decides when to release. The episode ends when the ball goes through the rim, misses, or time runs out.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

from . import mdp

HOOP_USD = os.path.join(os.path.dirname(__file__), "assets", "hoop.usda")

# Ball center at Franka's default joint pose: 8.5 cm along the hand axis (mid finger pad), from forward kinematics.
BALL_START = (0.450, 0.0, 0.398)
# Nominal hoop position (rim center), randomized on reset by EventCfg.reset_hoop.
HOOP_POS = (1.4, 0.0, 0.45)

# Stiffer arm so it can actually swing; gravity off on the links so PD targets are tracked cleanly.
ROBOT_CFG = FRANKA_PANDA_CFG.copy()  # deep copy: we mutate nested fields below
ROBOT_CFG.prim_path = "{ENV_REGEX_NS}/Robot"
ROBOT_CFG.spawn.rigid_props.disable_gravity = True
ROBOT_CFG.init_state.joint_pos["panda_finger_joint.*"] = mdp.BALL_RADIUS
for name in ("panda_shoulder", "panda_forearm"):
    ROBOT_CFG.actuators[name].stiffness = 400.0
    ROBOT_CFG.actuators[name].damping = 40.0


@configclass
class ShootSceneCfg(InteractiveSceneCfg):
    robot = ROBOT_CFG

    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        visualizer_cfg=FRAME_MARKER_CFG.replace(prim_path="/Visuals/EEFrame"),
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand", name="end_effector", offset=OffsetCfg(pos=(0.0, 0.0, 0.1034))
            )
        ],
    )

    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        init_state=RigidObjectCfg.InitialStateCfg(pos=BALL_START),
        spawn=sim_utils.SphereCfg(
            radius=mdp.BALL_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16, max_depenetration_velocity=1.0, max_linear_velocity=50.0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.2, dynamic_friction=1.0, restitution=0.6
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.35, 0.05)),
        ),
    )

    hoop = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Hoop",
        init_state=RigidObjectCfg.InitialStateCfg(pos=HOOP_POS),
        spawn=UsdFileCfg(
            usd_path=HOOP_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        ),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.5, 0.0, 0.0), rot=(0.707, 0.0, 0.0, 0.707)),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light", spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0)
    )


@configclass
class ActionsCfg:
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=["panda_joint.*"], scale=1.0, use_default_offset=True
    )
    gripper_action = mdp.ReleaseActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
        min_hold_time=0.3,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        ball_pos = ObsTerm(func=mdp.position_in_robot_frame, params={"asset_cfg": SceneEntityCfg("ball")})
        ball_vel = ObsTerm(func=mdp.root_lin_vel_w, params={"asset_cfg": SceneEntityCfg("ball")})
        hoop_pos = ObsTerm(func=mdp.position_in_robot_frame, params={"asset_cfg": SceneEntityCfg("hoop")})
        released = ObsTerm(func=mdp.is_released)
        time = ObsTerm(func=mdp.episode_time)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_hoop = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.2, 0.3), "y": (-0.4, 0.4), "z": (-0.15, 0.15)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("hoop"),
        },
    )


@configclass
class RewardsCfg:
    ball_near_hoop = RewTerm(func=mdp.ball_near_hoop, params={"std": 0.4}, weight=1.0)
    made_shot = RewTerm(func=mdp.made_shot, weight=50.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-3)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    made_shot = DoneTerm(func=mdp.ball_through_hoop)
    missed = DoneTerm(func=mdp.ball_missed)


@configclass
class ShootEnvCfg(ManagerBasedRLEnvCfg):
    scene: ShootSceneCfg = ShootSceneCfg(num_envs=4096, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 2.5
        self.sim.dt = 0.01  # 100 Hz physics, 50 Hz policy
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
        # camera looking at the robot and the hoop from the side
        self.viewer.eye = (0.8, -2.6, 1.0)
        self.viewer.lookat = (0.9, 0.0, 0.3)


@configclass
class ShootEnvCfg_PLAY(ShootEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 9
        self.scene.env_spacing = 4.0
        # keep the camera on env 0 for recordings
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0
