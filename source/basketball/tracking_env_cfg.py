"""Floating-base G1 whole-body tracking of a retargeted human shot (no ball yet).

Robot: Isaac Sim's 29-DoF G1 (floating base). PD gains follow BeyondMimic: per motor type,
kp = armature * w^2 and kd = 2 * zeta * armature * w with w = 2*pi*10 Hz and zeta = 2. Effort / velocity limits
come from Unitree's g1_29dof URDF. The action scale per joint is 0.25 * effort_limit / kp.
"""

import math
import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from . import tracking_mdp as mdp

MOTION_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "motions")

# motor armatures (kg m^2) from Unitree's motor specs, as used by BeyondMimic
ARM_5020 = 0.003609725
ARM_7520_14 = 0.010177520
ARM_7520_22 = 0.025101925
ARM_4010 = 0.00425
ARM_ANKLE = 2 * ARM_5020  # ankle and waist roll/pitch use two 5020 motors through a linkage
W = 2 * math.pi * 10.0
ZETA = 2.0


def _gains(armature):
    return armature * W**2, 2 * ZETA * armature * W


# joint group -> (regex, armature, effort limit Nm, velocity limit rad/s)
JOINT_GROUPS = {
    "hip_pitch_yaw": ([".*_hip_pitch_joint", ".*_hip_yaw_joint"], ARM_7520_14, 88.0, 32.0),
    "hip_roll": ([".*_hip_roll_joint"], ARM_7520_22, 88.0, 32.0),
    "knee": ([".*_knee_joint"], ARM_7520_22, 139.0, 20.0),
    "ankle": ([".*_ankle_pitch_joint", ".*_ankle_roll_joint"], ARM_ANKLE, 35.0, 30.0),
    "waist_yaw": (["waist_yaw_joint"], ARM_7520_14, 88.0, 32.0),
    "waist_roll_pitch": (["waist_roll_joint", "waist_pitch_joint"], ARM_ANKLE, 35.0, 30.0),
    "arm": ([".*_shoulder_pitch_joint", ".*_shoulder_roll_joint", ".*_shoulder_yaw_joint", ".*_elbow_joint",
             ".*_wrist_roll_joint"], ARM_5020, 25.0, 37.0),
    "wrist": ([".*_wrist_pitch_joint", ".*_wrist_yaw_joint"], ARM_4010, 5.0, 22.0),
}

BODY_JOINT_EXPRS = [e for exprs, *_ in JOINT_GROUPS.values() for e in exprs]

ACTUATORS = {
    # Dex3 three-finger hands: held in their default pose for now (not part of the action space)
    "hands": ImplicitActuatorCfg(joint_names_expr=[".*_hand_.*"], effort_limit_sim=2.45, velocity_limit_sim=10.0,
                                 stiffness=4.0, damping=0.2, armature=0.001),
}
ACTION_SCALE = {}
for name, (exprs, armature, effort, vel) in JOINT_GROUPS.items():
    kp, kd = _gains(armature)
    ACTUATORS[name] = ImplicitActuatorCfg(
        joint_names_expr=exprs, effort_limit_sim=effort, velocity_limit_sim=vel, stiffness=kp, damping=kd,
        armature=armature,
    )
    for e in exprs:
        ACTION_SCALE[e] = 0.25 * effort / kp

G1_TRACKING_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False, retain_accelerations=False, linear_damping=0.0, angular_damping=0.0,
            max_linear_velocity=1000.0, max_angular_velocity=1000.0, max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, fix_root_link=False, solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.76),
        joint_pos={
            ".*_hip_pitch_joint": -0.312, ".*_knee_joint": 0.669, ".*_ankle_pitch_joint": -0.363,
            ".*_elbow_joint": 0.6, "left_shoulder_roll_joint": 0.2, "right_shoulder_roll_joint": -0.2,
            ".*_shoulder_pitch_joint": 0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators=ACTUATORS,
)

TRACKED_BODIES = [
    "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link", "right_hip_roll_link",
    "right_knee_link", "right_ankle_roll_link", "torso_link", "left_shoulder_roll_link", "left_elbow_link",
    "left_wrist_yaw_link", "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
]
END_EFFECTORS = ["left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]


@configclass
class TrackingSceneCfg(InteractiveSceneCfg):
    robot = G1_TRACKING_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    floor = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=GroundPlaneCfg(
            color=(0.72, 0.52, 0.32),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
        ),
    )
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))


@configclass
class CommandsCfg:
    motion = mdp.MotionCommandCfg(
        motion_file=os.path.join(MOTION_DIR, "g1_124_04.npz"), body_names=TRACKED_BODIES, debug_vis=False
    )


@configclass
class ActionsCfg:
    joints = mdp.JointPositionActionCfg(asset_name="robot", joint_names=BODY_JOINT_EXPRS, scale=ACTION_SCALE,
                                        use_default_offset=True)


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        anchor_pos = ObsTerm(func=mdp.motion_anchor_pos_b, noise=Unoise(n_min=-0.25, n_max=0.25))
        anchor_ori = ObsTerm(func=mdp.motion_anchor_ori_b, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.5, n_max=0.5))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        anchor_pos = ObsTerm(func=mdp.motion_anchor_pos_b)
        anchor_ori = ObsTerm(func=mdp.motion_anchor_ori_b)
        body_pos = ObsTerm(func=mdp.robot_body_pos_b)
        phase = ObsTerm(func=mdp.motion_phase)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    floor_friction = EventTerm(
        func=mdp.randomize_rigid_body_material, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "static_friction_range": (0.3, 1.6),
                "dynamic_friction_range": (0.3, 1.2), "restitution_range": (0.0, 0.5), "num_buckets": 64},
    )
    torso_com = EventTerm(
        func=mdp.randomize_rigid_body_com, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
                "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)}},
    )
    push = EventTerm(
        func=mdp.push_by_setting_velocity, mode="interval", interval_range_s=(1.0, 3.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.2, 0.2), "roll": (-0.52, 0.52),
                                   "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78)}},
    )


@configclass
class RewardsCfg:
    anchor_pos = RewTerm(func=mdp.anchor_pos_exp, weight=0.5, params={"std": 0.3})
    anchor_ori = RewTerm(func=mdp.anchor_ori_exp, weight=0.5, params={"std": 0.4})
    body_pos = RewTerm(func=mdp.body_pos_exp, weight=1.0, params={"std": 0.3})
    body_ori = RewTerm(func=mdp.body_ori_exp, weight=1.0, params={"std": 0.4})
    body_lin_vel = RewTerm(func=mdp.body_lin_vel_exp, weight=1.0, params={"std": 1.0})
    body_ang_vel = RewTerm(func=mdp.body_ang_vel_exp, weight=1.0, params={"std": 3.14})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    # body joints only: the Dex3 finger joints rest at 0, which is at their hard limit and outside the soft limits,
    # so including them charged a constant penalty from the first step and the policy learned to fall immediately
    joint_limit = RewTerm(func=mdp.joint_pos_limits, weight=-10.0,
                          params={"asset_cfg": SceneEntityCfg("robot", joint_names=BODY_JOINT_EXPRS)})


@configclass
class TerminationsCfg:
    motion_end = DoneTerm(func=mdp.motion_finished, time_out=True)
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_height = DoneTerm(func=mdp.anchor_height_error, params={"threshold": 0.25})
    anchor_tilt = DoneTerm(func=mdp.anchor_tilt_error, params={"threshold": 0.8})
    end_effectors = DoneTerm(func=mdp.end_effector_height_error,
                             params={"threshold": 0.25, "body_names": END_EFFECTORS})


@configclass
class G1TrackingEnvCfg(ManagerBasedRLEnvCfg):
    scene: TrackingSceneCfg = TrackingSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.floor.spawn.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0
        self.viewer.eye = (2.5, -2.5, 1.4)
        self.viewer.lookat = (0.0, 0.0, 0.8)


@configclass
class G1TrackingEnvCfg_PLAY(G1TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.observations.policy.enable_corruption = False
        self.events.push = None
        # always play the whole clip from the first frame, without start noise
        self.commands.motion.random_start = False
        self.commands.motion.pose_noise = 0.0
        self.commands.motion.vel_noise = 0.0
        self.commands.motion.joint_noise = 0.0
        self.viewer.eye = (0.6, -3.2, 1.2)
        self.viewer.lookat = (0.4, 0.0, 0.85)
