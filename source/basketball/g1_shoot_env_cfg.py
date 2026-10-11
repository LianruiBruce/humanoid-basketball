"""Unitree G1 (Inspire hands) shoots a regulation basketball at a regulation NBA hoop.

v1: fixed pelvis, upper-body shot. The ball starts squeezed between the palms; the policy outputs joint targets
for both arms (offsets around the holding pose) and the ball leaves the hands only through the arm motion.
Arm motors use Unitree's official limits (25 Nm / 37 rad/s; wrist pitch & yaw 5 Nm / 22 rad/s) with gravity on.
A curriculum moves the hoop from 1.5 m out to the three-point line as the make rate improves.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_assets.robots.unitree import G1_INSPIRE_FTP_CFG

from . import amp
from . import g1_mdp as mdp

HOOP_USD = os.path.join(os.path.dirname(__file__), "assets", "nba_hoop.usda")
ARM_JOINTS = [".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_.*_joint"]
PELVIS_HEIGHT = 0.793  # feet on the floor with straight legs
# Holding pose: shoulders rolled inward 0.3 rad squeeze the ball between the palms (see scripts/hold_test_g1.py).
HOLD_OFFSET = {"left_shoulder_roll_joint": -0.3, "right_shoulder_roll_joint": 0.3}
BALL_START = (0.30, 0.0, 0.893)
HOOP_START = (2.0, 0.0, mdp.RIM_HEIGHT)

ROBOT_CFG = G1_INSPIRE_FTP_CFG.copy()
ROBOT_CFG.prim_path = "{ENV_REGEX_NS}/Robot"
ROBOT_CFG.spawn.rigid_props.disable_gravity = False
ROBOT_CFG.spawn.articulation_props.fix_root_link = True
ROBOT_CFG.init_state = ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, PELVIS_HEIGHT),
    # start just short of contact; the PD targets (HOLD_OFFSET) then squeeze the ball
    joint_pos={
        "left_shoulder_roll_joint": -0.15,
        "right_shoulder_roll_joint": 0.15,
        ".*_(index|middle|ring|pinky)_proximal_joint": 0.4,
        ".*_thumb_proximal_yaw_joint": 0.4,
    },
)
ROBOT_CFG.actuators["arms"] = ImplicitActuatorCfg(
    joint_names_expr=[".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_roll_joint"],
    effort_limit_sim=25.0,
    velocity_limit_sim=37.0,
    stiffness=400.0,
    damping=20.0,
    armature=0.01,
)
ROBOT_CFG.actuators["wrists"] = ImplicitActuatorCfg(
    joint_names_expr=[".*_wrist_pitch_joint", ".*_wrist_yaw_joint"],
    effort_limit_sim=5.0,
    velocity_limit_sim=22.0,
    stiffness=100.0,
    damping=5.0,
    armature=0.01,
)


@configclass
class G1ShootSceneCfg(InteractiveSceneCfg):
    robot = ROBOT_CFG

    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        init_state=RigidObjectCfg.InitialStateCfg(pos=BALL_START),
        spawn=sim_utils.SphereCfg(
            radius=mdp.BALL_RADIUS,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0, max_linear_velocity=40.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.62),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0, dynamic_friction=0.8, restitution=0.8
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.33, 0.05), roughness=0.7),
        ),
    )

    hoop = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Hoop",
        init_state=RigidObjectCfg.InitialStateCfg(pos=HOOP_START),
        spawn=UsdFileCfg(
            usd_path=HOOP_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True),
        ),
    )

    floor = AssetBaseCfg(prim_path="/World/GroundPlane", spawn=GroundPlaneCfg(color=(0.72, 0.52, 0.32)))

    dome = AssetBaseCfg(
        prim_path="/World/dome", spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=1500.0)
    )
    sun = AssetBaseCfg(
        prim_path="/World/sun",
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.82, 0.25, 0.42, 0.3)),
        spawn=sim_utils.DistantLightCfg(intensity=2500.0, angle=1.0),
    )


@configclass
class ActionsCfg:
    arms = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=ARM_JOINTS, scale=1.0, use_default_offset=False, offset=HOLD_OFFSET
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        arm_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)})
        arm_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)})
        ball_pos = ObsTerm(func=mdp.position_in_robot_frame, params={"asset_cfg": SceneEntityCfg("ball")})
        ball_vel = ObsTerm(func=mdp.root_lin_vel_w, params={"asset_cfg": SceneEntityCfg("ball")})
        ball_to_palms = ObsTerm(func=mdp.ball_to_palms)
        hoop_pos = ObsTerm(func=mdp.position_in_robot_frame, params={"asset_cfg": SceneEntityCfg("hoop")})
        time = ObsTerm(func=mdp.episode_time)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    # places the hoop and runs the distance curriculum (half frontier shots, half review of learned distances)
    reset_hoop = EventTerm(
        func=mdp.reset_hoop_curriculum,
        mode="reset",
        params={"start_distance": 1.5, "max_distance": 6.75, "step": 0.25, "promote_at": 0.4, "spread": 0.25,
                "frontier_prob": 0.5, "lateral": 0.3},
    )


@configclass
class RewardsCfg:
    ball_near_hoop = RewTerm(func=mdp.ball_near_hoop, params={"std": 1.0}, weight=1.0)
    made_shot = RewTerm(func=mdp.made_shot, weight=100.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    made_shot = DoneTerm(func=mdp.ball_through_hoop, params={"rim_radius": mdp.RIM_RADIUS})
    missed = DoneTerm(func=mdp.ball_missed)


@configclass
class CurriculumCfg:
    distance = CurrTerm(func=mdp.hoop_curriculum_state)  # logging only; the logic lives in reset_hoop


@configclass
class G1ShootEnvCfg(ManagerBasedRLEnvCfg):
    scene: G1ShootSceneCfg = G1ShootSceneCfg(num_envs=4096, env_spacing=12.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 3.0
        self.sim.dt = 0.005  # 200 Hz physics, 50 Hz policy
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 32 * 1024
        self.viewer.origin_type = "env"
        self.viewer.env_index = 0
        self.viewer.eye = (-3.6, -3.4, 2.3)
        self.viewer.lookat = (1.0, 0.0, 1.9)


@configclass
class G1ShootEnvCfg_PLAY(G1ShootEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4


@configclass
class FormRewardsCfg(RewardsCfg):
    """Adds shooting-form terms on top of make/miss: release over the head with a proper arc, ball kept on the
    shooting line with both hands level, steep entry into the rim."""

    release_height = RewTerm(func=mdp.release_form, params={"component": "height"}, weight=30.0)
    release_angle = RewTerm(func=mdp.release_form, params={"component": "angle"}, weight=30.0)
    entry_angle = RewTerm(func=mdp.entry_angle_bonus, weight=50.0)
    dropped = RewTerm(func=mdp.terminated_by, params={"term_name": "dropped"}, weight=-50.0)
    # per-step posture penalties start at 0 and are switched on by the curriculum once the policy can shoot;
    # with them on from step 0 the policy learned to dump the ball immediately to end the episode
    off_line = RewTerm(func=mdp.ball_off_shooting_line, weight=0.0)
    hand_asymmetry = RewTerm(func=mdp.palm_height_asymmetry, weight=0.0)


@configclass
class FormTerminationsCfg(TerminationsCfg):
    dropped = DoneTerm(func=mdp.ball_dropped)


@configclass
class FormCurriculumCfg(CurriculumCfg):
    off_line_on = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "off_line", "weight": -3.0, "num_steps": 24_000}
    )
    hand_asymmetry_on = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "hand_asymmetry", "weight": -3.0, "num_steps": 24_000}
    )


@configclass
class G1ShootFormEnvCfg(G1ShootEnvCfg):
    rewards: FormRewardsCfg = FormRewardsCfg()
    terminations: FormTerminationsCfg = FormTerminationsCfg()
    curriculum: FormCurriculumCfg = FormCurriculumCfg()


@configclass
class G1ShootFormEnvCfg_PLAY(G1ShootFormEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4


##
# Stage 3: human-like form from motion capture (Adversarial Motion Prior)
##

MOCAP_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "mocap")
AMP_REFERENCES = [
    os.path.join(MOCAP_DIR, f) for f in ("g1_124_03_100hz.npz", "g1_124_04_100hz.npz", "g1_124_05_100hz.npz")
]


@configclass
class AMPRewardsCfg(FormRewardsCfg):
    # CMU subject 124 shot / free throw / jump shot, retargeted to the G1 arms
    amp_style = RewTerm(func=amp.amp_style_reward, params={"reference_files": AMP_REFERENCES}, weight=2.0)
    pose_match = RewTerm(func=amp.pose_match_reward, params={"reference_files": AMP_REFERENCES, "scale": 5.0}, weight=2.0)


@configclass
class AMPCurriculumCfg(FormCurriculumCfg):
    amp = CurrTerm(func=amp.amp_stats)  # logging only


@configclass
class G1ShootAMPEnvCfg(G1ShootFormEnvCfg):
    rewards: AMPRewardsCfg = AMPRewardsCfg()
    curriculum: AMPCurriculumCfg = AMPCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        # fine-tuned from the form policy, which already shoots out to the 3-point line
        self.events.reset_hoop.params["initial_frontier"] = 6.75


@configclass
class G1ShootAMPEnvCfg_PLAY(G1ShootAMPEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4


##
# Domain randomization: ball, robot dynamics, control latency, sensor noise
##

ARM_BODIES = [".*_shoulder_.*_link", ".*_elbow_link", ".*_wrist_.*_link"]
HAND_BODIES = ["L_.*", "R_.*", ".*_wrist_yaw_link"]


@configclass
class DREventCfg(EventCfg):
    # ball: NBA allows 567-650 g; friction / bounce vary with the ball and its wear
    ball_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass, mode="startup",
        params={"asset_cfg": SceneEntityCfg("ball"), "mass_distribution_params": (0.567, 0.650), "operation": "abs"},
    )
    ball_material = EventTerm(
        func=mdp.randomize_rigid_body_material, mode="startup",
        params={"asset_cfg": SceneEntityCfg("ball"), "static_friction_range": (0.7, 1.2),
                "dynamic_friction_range": (0.6, 1.0), "restitution_range": (0.70, 0.88), "num_buckets": 64},
    )
    # robot: link masses, motor strength (PD gains), joint friction, palm friction
    arm_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=ARM_BODIES), "mass_distribution_params": (0.9, 1.1),
                "operation": "scale"},
    )
    arm_gains = EventTerm(
        func=mdp.randomize_actuator_gains, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS),
                "stiffness_distribution_params": (0.8, 1.2), "damping_distribution_params": (0.8, 1.2),
                "operation": "scale"},
    )
    arm_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS),
                "friction_distribution_params": (0.0, 0.05), "operation": "abs"},
    )
    hand_material = EventTerm(
        func=mdp.randomize_rigid_body_material, mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=HAND_BODIES), "static_friction_range": (0.6, 1.2),
                "dynamic_friction_range": (0.5, 1.0), "restitution_range": (0.0, 0.1), "num_buckets": 64},
    )


@configclass
class G1ShootDREnvCfg(G1ShootAMPEnvCfg):
    events: DREventCfg = DREventCfg()

    def __post_init__(self):
        super().__post_init__()
        # 0 / 5 / 10 ms control latency (physics substeps), resampled every episode
        self.actions.arms = mdp.DelayedJointPositionActionCfg(
            asset_name="robot", joint_names=ARM_JOINTS, scale=1.0, use_default_offset=False, offset=HOLD_OFFSET,
            max_delay_substeps=2,
        )
        # sensor noise (hoop noise stands in for perception error until the camera stage)
        obs = self.observations.policy
        obs.enable_corruption = True
        obs.arm_pos.noise = Unoise(n_min=-0.01, n_max=0.01)
        obs.arm_vel.noise = Unoise(n_min=-0.2, n_max=0.2)
        obs.ball_pos.noise = Unoise(n_min=-0.01, n_max=0.01)
        obs.ball_vel.noise = Unoise(n_min=-0.1, n_max=0.1)
        obs.ball_to_palms.noise = Unoise(n_min=-0.01, n_max=0.01)
        obs.hoop_pos.noise = Unoise(n_min=-0.03, n_max=0.03)


@configclass
class G1ShootDREnvCfg_PLAY(G1ShootDREnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4


##
# Robust sim2real setup: observation history for the actor, privileged critic (asymmetric actor-critic)
##


@configclass
class RobustObservationsCfg(ObservationsCfg):
    @configclass
    class CriticCfg(ObsGroup):
        """Noise-free state plus the hidden randomization parameters. Used only by the value function."""

        arm_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)})
        arm_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS)})
        ball_pos = ObsTerm(func=mdp.position_in_robot_frame, params={"asset_cfg": SceneEntityCfg("ball")})
        ball_vel = ObsTerm(func=mdp.root_lin_vel_w, params={"asset_cfg": SceneEntityCfg("ball")})
        ball_to_palms = ObsTerm(func=mdp.ball_to_palms)
        hoop_pos = ObsTerm(func=mdp.position_in_robot_frame, params={"asset_cfg": SceneEntityCfg("hoop")})
        time = ObsTerm(func=mdp.episode_time)
        actions = ObsTerm(func=mdp.last_action)
        dynamics = ObsTerm(func=mdp.privileged_dynamics, params={"arm_joints": ARM_JOINTS})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()


@configclass
class RobustCurriculumCfg(AMPCurriculumCfg):
    # Style rewards are per-step and the holding pose already looks like the reference, so from scratch the
    # policy learned to hold the ball and farm them (episodes grew 12 -> 110 steps, makes fell). Switch them on
    # only after ~1500 iterations (32 env steps each), once the policy already shoots.
    amp_style_on = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "amp_style", "weight": 2.0, "num_steps": 48_000}
    )
    pose_match_on = CurrTerm(
        func=mdp.modify_reward_weight, params={"term_name": "pose_match", "weight": 2.0, "num_steps": 48_000}
    )


@configclass
class G1ShootRobustEnvCfg(G1ShootDREnvCfg):
    observations: RobustObservationsCfg = RobustObservationsCfg()
    curriculum: RobustCurriculumCfg = RobustCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.rewards.amp_style.weight = 0.0
        self.rewards.pose_match.weight = 0.0
        # actor sees the last 3 noisy observations, so latency and motor strength can be inferred
        self.observations.policy.history_length = 3
        # trained from scratch: distance curriculum starts close again; promote at a realistic make rate
        self.events.reset_hoop.params["initial_frontier"] = None
        self.events.reset_hoop.params["promote_at"] = 0.3


@configclass
class G1ShootRobustFullRangeEnvCfg(G1ShootRobustEnvCfg):
    """Fine-tuning stage: under full randomization the from-scratch curriculum stalled at 5.25 m, so the
    3-point range was never trained. Sample the whole 1.5-6.75 m range from the start, and add the ballistic aim
    reward: sampling alone was not enough (3PT stayed at 0%, the policy just pushed the ball a meter forward)."""

    def __post_init__(self):
        super().__post_init__()
        self.events.reset_hoop.params["initial_frontier"] = 6.75
        self.rewards.ballistic_aim = RewTerm(func=mdp.ballistic_aim, params={"std_fraction": 0.3}, weight=150.0)


##
# Stage 3b: right-handed one-hand shot (guide hand stays on the side of the ball)
##

ONE_HAND_REFERENCES = [
    os.path.join(MOCAP_DIR, f)
    for f in ("g1_124_03_onehand_100hz.npz", "g1_124_04_onehand_100hz.npz", "g1_124_05_onehand_100hz.npz")
]


@configclass
class G1ShootOneHandEnvCfg(G1ShootRobustFullRangeEnvCfg):
    """Fine-tunes the robust policy toward one-hand form: same task and randomization, motion prior swapped to
    references whose left (guide) arm freezes at the set point (``retarget_cmu_to_g1.py --one_hand``)."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.amp_style.params["reference_files"] = ONE_HAND_REFERENCES
        self.rewards.pose_match.params["reference_files"] = ONE_HAND_REFERENCES
        # the policy already shoots: style rewards on from the first iteration
        self.rewards.amp_style.weight = 2.0
        self.rewards.pose_match.weight = 2.0
        self.curriculum.amp_style_on = None
        self.curriculum.pose_match_on = None


@configclass
class G1ShootOneHandEnvCfg_PLAY(G1ShootOneHandEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4


@configclass
class G1ShootRobustEnvCfg_PLAY(G1ShootRobustEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
