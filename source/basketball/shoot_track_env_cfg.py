"""Whole-body shooting: the G1 tracks a retargeted human shot (free throw or jump shot) on an NBA court, holding a
size-7 ball in its Dex3 hands, and is rewarded for making the shot.

Tracking rewards keep the motion human-like; the shot terms (ballistic aim at release, make bonus, drop penalty)
let the arms deviate from the reference where needed: the retargeted human motion is too slow to reach the rim on
its own, so wrists are excluded from the fall terminations.
"""

import os

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from . import tracking_mdp as mdp
from .tracking_env_cfg import (
    MOTION_DIR,
    TRACKED_BODIES,
    G1TrackingEnvCfg,
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
    TrackingSceneCfg,
)

HOOP_USD = os.path.join(os.path.dirname(__file__), "assets", "nba_hoop.usda")
FEET = ["left_ankle_roll_link", "right_ankle_roll_link"]


@configclass
class ShootSceneCfg(TrackingSceneCfg):
    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.0, 1.0)),
        spawn=sim_utils.SphereCfg(
            radius=0.1193,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0, max_linear_velocity=40.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.62),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8,
                                                            restitution=0.8),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.33, 0.05), roughness=0.7),
        ),
    )
    hoop = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Hoop",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(4.2, 0.0, 3.05)),
        spawn=UsdFileCfg(usd_path=HOOP_USD,
                         rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True, disable_gravity=True)),
    )


@configclass
class ShootCommandsCfg:
    motion = mdp.BallMotionCommandCfg(
        motion_file=os.path.join(MOTION_DIR, "g1_124_04_ball.npz"), body_names=TRACKED_BODIES, debug_vis=False,
        shot_distance=4.19,
    )


@configclass
class ShootObservationsCfg(ObservationsCfg):
    def __post_init__(self):
        for group in (self.policy, self.critic):
            group.ball_pos = ObsTerm(func=mdp.ball_rel_anchor, params={"asset_name": "ball"})
            group.ball_vel = ObsTerm(func=mdp.ball_vel)
            group.hoop_pos = ObsTerm(func=mdp.ball_rel_anchor, params={"asset_name": "hoop"})
        self.policy.phase = ObsTerm(func=mdp.motion_phase)


@configclass
class ShootRewardsCfg(RewardsCfg):
    ballistic_aim = RewTerm(func=mdp.ballistic_aim, weight=200.0, params={"std_fraction": 0.3})
    made = RewTerm(func=mdp.terminated_by, weight=500.0, params={"term_name": "made"})
    dropped = RewTerm(func=mdp.terminated_by, weight=-200.0, params={"term_name": "dropped"})
    ball_near_hoop = RewTerm(func=mdp.ball_near_hoop, weight=2.0, params={"std": 1.0})
    hold = RewTerm(func=mdp.hold_ball, weight=2.0, params={"std": 0.1})


@configclass
class ShootTerminationsCfg(TerminationsCfg):
    made = DoneTerm(func=mdp.ball_through_rim)
    missed = DoneTerm(func=mdp.ball_missed)
    dropped = DoneTerm(func=mdp.ball_dropped)

    def __post_init__(self):
        self.motion_end = None  # keep going after the clip ends (the reference holds its last frame) until the
        #                         ball resolves or time runs out
        self.end_effectors.params["body_names"] = FEET  # wrists may leave the reference to put power on the ball


@configclass
class G1ShootTrackEnvCfg(G1TrackingEnvCfg):
    scene: ShootSceneCfg = ShootSceneCfg(num_envs=4096, env_spacing=12.0)
    commands: ShootCommandsCfg = ShootCommandsCfg()
    observations: ShootObservationsCfg = ShootObservationsCfg()
    rewards: ShootRewardsCfg = ShootRewardsCfg()
    terminations: ShootTerminationsCfg = ShootTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 3.5
        self.events.push = None  # no shoves while shooting
        # Dex3 fingers curled around the ball (signs follow each joint's range)
        jp = dict(self.scene.robot.init_state.joint_pos)
        jp.update({
            "left_hand_index_.*": -0.5, "left_hand_middle_.*": -0.5, "right_hand_index_.*": 0.5,
            "right_hand_middle_.*": 0.5, "left_hand_thumb_2_joint": 0.4, "right_hand_thumb_2_joint": -0.4,
        })
        self.scene.robot = self.scene.robot.replace(init_state=self.scene.robot.init_state.replace(joint_pos=jp))
        self.scene.robot.actuators["hands"].stiffness = 10.0
        self.scene.robot.actuators["hands"].damping = 0.5
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 32 * 1024
        self.viewer.eye = (-2.5, -3.0, 2.0)
        self.viewer.lookat = (2.0, 0.0, 1.8)


@configclass
class G1ShootTrackEnvCfg_PLAY(G1ShootTrackEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.observations.policy.enable_corruption = False
        self.commands.motion.random_start = False


@configclass
class G1JumpShotEnvCfg(G1ShootTrackEnvCfg):
    """Jump shot (CMU 124_05): the shooter moves ~0.9 m forward into the jump; rim 4.5 m ahead of the release."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = os.path.join(MOTION_DIR, "g1_124_05_ball.npz")
        self.commands.motion.shot_distance = 4.5


@configclass
class G1JumpShotEnvCfg_PLAY(G1JumpShotEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.observations.policy.enable_corruption = False
        self.commands.motion.random_start = False
