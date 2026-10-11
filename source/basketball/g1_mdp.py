"""MDP terms for the G1 humanoid shooting a regulation basketball.

There is no "release" button: the ball sits between the palms and leaves the hands only through the arm motion.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils import configclass

from .mdp import ball_through_hoop, episode_time, position_in_robot_frame  # noqa: F401

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

BALL_RADIUS = 0.1193  # NBA size 7
RIM_RADIUS = 0.2286
RIM_HEIGHT = 3.05
PALM_BODIES = ("L_middle_proximal", "R_middle_proximal")


def _palm_center(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    ids = [robot.body_names.index(n) for n in PALM_BODIES]
    return robot.data.body_pos_w[:, ids].mean(dim=1)


def ball_released(
    env: ManagerBasedRLEnv,
    threshold: float = 0.25,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Ball center is farther than ``threshold`` from the midpoint between the palms."""
    ball = env.scene[ball_cfg.name].data.root_pos_w
    return torch.norm(ball - _palm_center(env, robot_cfg), dim=1) > threshold


##
# Actions
##


class DelayedJointPositionAction(JointPositionAction):
    """Joint position targets that reach the motors after a per-env latency of 0..``max_delay_substeps`` physics
    substeps (5 ms each at 200 Hz), resampled every episode.

    A whole-policy-step (20 ms) delay was tried first: it is coarser and larger than the G1's real ~5-10 ms
    latency, and since the policy cannot observe it, it wrecked the release timing (make rate stalled near 9%).
    """

    cfg: DelayedJointPositionActionCfg

    def __init__(self, cfg: DelayedJointPositionActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._prev = self._offset * torch.ones_like(self._processed_actions)
        self._delay = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._substep = 0
        # kept for logging / debugging scripts: True when the env has any latency
        self._delayed = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def process_actions(self, actions: torch.Tensor):
        self._prev = self._processed_actions.clone()
        super().process_actions(actions)
        self._substep = 0

    def apply_actions(self):
        late = self._delay > self._substep  # still within this env's latency window
        target = torch.where(late[:, None], self._prev, self._processed_actions)
        self._asset.set_joint_position_target(target, joint_ids=self._joint_ids)
        self._substep += 1

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._prev[env_ids] = (self._offset * torch.ones_like(self._processed_actions))[env_ids]
        self._processed_actions[env_ids] = self._prev[env_ids]
        n = self._delay[env_ids].shape[0]
        self._delay[env_ids] = torch.randint(0, self.cfg.max_delay_substeps + 1, (n,), device=self.device)
        self._delayed[env_ids] = self._delay[env_ids] > 0


@configclass
class DelayedJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = DelayedJointPositionAction
    max_delay_substeps: int = 2


##
# Observations
##


def ball_to_palms(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return env.scene[ball_cfg.name].data.root_pos_w - _palm_center(env, robot_cfg)


class privileged_dynamics(ManagerTermBase):
    """Critic-only observation: the hidden domain-randomization parameters of this env.

    [ball mass / nominal, control delay / max, mean arm stiffness / nominal, mean arm damping / nominal,
    mean arm joint friction]. Ball mass is read once (it is randomized at startup) and cached.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        robot = env.scene["robot"]
        self._arm_ids, _ = robot.find_joints(cfg.params["arm_joints"])
        self._nominal_kp = robot.data.default_joint_stiffness[:, self._arm_ids].clone()
        self._nominal_kd = robot.data.default_joint_damping[:, self._arm_ids].clone()
        self._ball_mass = None

    def __call__(self, env: ManagerBasedRLEnv, arm_joints: list[str], nominal_ball_mass: float = 0.62,
                 action_name: str = "arms") -> torch.Tensor:
        robot = env.scene["robot"]
        if self._ball_mass is None:
            self._ball_mass = env.scene["ball"].root_physx_view.get_masses()[:, 0].to(env.device) / nominal_ball_mass
        act = env.action_manager.get_term(action_name)
        max_delay = max(getattr(act.cfg, "max_delay_substeps", 1), 1)
        delay = getattr(act, "_delay", torch.zeros(env.num_envs, device=env.device)).float() / max_delay
        kp = (robot.data.joint_stiffness[:, self._arm_ids] / self._nominal_kp).mean(dim=1)
        kd = (robot.data.joint_damping[:, self._arm_ids] / self._nominal_kd).mean(dim=1)
        fric = getattr(robot.data, "joint_friction_coeff", torch.zeros_like(robot.data.joint_pos))
        fric = fric[:, self._arm_ids].mean(dim=1)
        return torch.stack([self._ball_mass, delay, kp, kd, fric], dim=1)


##
# Terminations
##


def ball_missed(
    env: ManagerBasedRLEnv,
    margin: float = 0.4,
    floor_height: float = 0.35,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    hoop_cfg: SceneEntityCfg = SceneEntityCfg("hoop"),
) -> torch.Tensor:
    """Released ball is falling and already below the rim, or the ball is (nearly) on the floor."""
    ball = env.scene[ball_cfg.name].data
    rim_z = env.scene[hoop_cfg.name].data.root_pos_w[:, 2]
    falling_below = (ball.root_lin_vel_w[:, 2] < 0) & (ball.root_pos_w[:, 2] < rim_z - margin)
    on_floor = ball.root_pos_w[:, 2] < env.scene.env_origins[:, 2] + floor_height
    return (ball_released(env) & falling_below) | on_floor


def ball_dropped(
    env: ManagerBasedRLEnv,
    near_radius: float = 1.0,
    max_height: float = 1.3,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Ball left the hands but is falling right in front of the robot: a drop, not a shot.

    Without this, dumping the ball is the cheapest way to end an episode (and dodge any per-step penalty).
    """
    ball = env.scene[ball_cfg.name].data
    root = env.scene[robot_cfg.name].data.root_pos_w
    near = torch.norm(ball.root_pos_w[:, :2] - root[:, :2], dim=1) < near_radius
    low = ball.root_pos_w[:, 2] - env.scene.env_origins[:, 2] < max_height
    return ball_released(env) & near & low & (ball.root_lin_vel_w[:, 2] < 0)


##
# Rewards
##


def terminated_by(env: ManagerBasedRLEnv, term_name: str) -> torch.Tensor:
    return env.termination_manager.get_term(term_name).float()


def ball_near_hoop(
    env: ManagerBasedRLEnv,
    std: float,
    aim_height: float = 0.25,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    hoop_cfg: SceneEntityCfg = SceneEntityCfg("hoop"),
) -> torch.Tensor:
    """While the ball is in flight: closeness to a point just above the rim center."""
    aim = env.scene[hoop_cfg.name].data.root_pos_w.clone()
    aim[:, 2] += aim_height
    dist = torch.norm(env.scene[ball_cfg.name].data.root_pos_w - aim, dim=1)
    return ball_released(env).float() * torch.exp(-dist / std)


def made_shot(env: ManagerBasedRLEnv, term_name: str = "made_shot") -> torch.Tensor:
    return env.termination_manager.get_term(term_name).float()


##
# Shooting-form rewards
##


class release_form(ManagerTermBase):
    """Style score measured when the ball leaves the hands, paid out only if that shot goes in.

    ``component="height"``: release point at or above ``height_ref`` (over the head, like a real shot).
    ``component="angle"``: launch angle close to ``angle_target`` degrees (a proper arc instead of a line drive).

    Paying at release made the bonus easier to farm than making shots (the policy optimized pretty releases and
    stopped aiming), so the score is held until the ``made_shot`` termination fires.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._released = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._score = torch.zeros(env.num_envs, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._released[env_ids] = False
        self._score[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        component: str,
        height_ref: float = 1.35,
        height_span: float = 0.35,
        angle_target: float = 52.0,
        angle_std: float = 12.0,
        term_name: str = "made_shot",
        ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    ) -> torch.Tensor:
        released = ball_released(env)
        just = released & ~self._released
        self._released |= released
        ball = env.scene[ball_cfg.name].data
        if component == "height":
            z = ball.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
            value = ((z - (height_ref - height_span)) / height_span).clamp(0.0, 1.0)
        else:
            v = ball.root_lin_vel_w
            ang = torch.rad2deg(torch.atan2(v[:, 2], torch.norm(v[:, :2], dim=1)))
            value = torch.exp(-(((ang - angle_target) / angle_std) ** 2))
        self._score = torch.where(just, value, self._score)
        return env.termination_manager.get_term(term_name).float() * self._score


class ballistic_aim(ManagerTermBase):
    """One-time reward at release: how close the ball's ballistic trajectory passes to the rim center.

    From the release position and velocity we solve for where the ball comes back down through rim height
    (no drag) and reward ``exp(-miss / std)``. If the trajectory never reaches rim height, the miss is the
    horizontal distance to the hoop plus the height shortfall. Unlike the distance-to-hoop shaping, this keeps a
    useful gradient for long shots: under full randomization the policy stopped trying from the 3-point line and
    just pushed the ball a meter forward, since every miss scored about the same.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._released = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._released[env_ids] = False

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std_fraction: float = 0.3,
        ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
        hoop_cfg: SceneEntityCfg = SceneEntityCfg("hoop"),
        robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """``std`` scales with the shot distance (a fixed 0.5 m std made a 6 m miss worth ~1e-5, i.e. no signal)."""
        released = ball_released(env)
        just = released & ~self._released
        self._released |= released
        p = env.scene[ball_cfg.name].data.root_pos_w
        v = env.scene[ball_cfg.name].data.root_lin_vel_w
        hoop = env.scene[hoop_cfg.name].data.root_pos_w
        g = 9.81
        dz = hoop[:, 2] - p[:, 2]
        disc = v[:, 2] ** 2 - 2.0 * g * dz
        reaches = disc >= 0
        t = (v[:, 2] + torch.sqrt(disc.clamp(min=0.0))) / g  # descending crossing of rim height
        landing = p[:, :2] + v[:, :2] * t.unsqueeze(1)
        miss = torch.norm(landing - hoop[:, :2], dim=1)
        apex = p[:, 2] + v[:, 2].clamp(min=0.0) ** 2 / (2.0 * g)
        short = torch.norm(hoop[:, :2] - p[:, :2], dim=1) + (hoop[:, 2] - apex).clamp(min=0.0)
        miss = torch.where(reaches, miss, short)
        shot_dist = torch.norm(hoop[:, :2] - env.scene[robot_cfg.name].data.root_pos_w[:, :2], dim=1)
        std = (std_fraction * shot_dist).clamp(min=0.3)
        return just.float() * torch.exp(-miss / std)


def ball_off_shooting_line(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    hoop_cfg: SceneEntityCfg = SceneEntityCfg("hoop"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """While holding: sideways distance of the ball from the robot->hoop line (no side-arm slings)."""
    p0 = env.scene[robot_cfg.name].data.root_pos_w[:, :2]
    d = env.scene[hoop_cfg.name].data.root_pos_w[:, :2] - p0
    d = d / torch.norm(d, dim=1, keepdim=True).clamp(min=1e-6)
    b = env.scene[ball_cfg.name].data.root_pos_w[:, :2] - p0
    lateral = torch.abs(b[:, 0] * d[:, 1] - b[:, 1] * d[:, 0])
    return (~ball_released(env)).float() * lateral


def palm_height_asymmetry(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """While holding: both hands at the same height (no one-arm fling)."""
    robot = env.scene[robot_cfg.name]
    ids = [robot.body_names.index(n) for n in PALM_BODIES]
    z = robot.data.body_pos_w[:, ids, 2]
    return (~ball_released(env)).float() * torch.abs(z[:, 0] - z[:, 1])


def entry_angle_bonus(
    env: ManagerBasedRLEnv,
    min_angle: float = 30.0,
    full_angle: float = 45.0,
    term_name: str = "made_shot",
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """On a make: reward a steep entry into the rim (NBA shooters enter around 45 deg)."""
    v = env.scene[ball_cfg.name].data.root_lin_vel_w
    entry = torch.rad2deg(torch.atan2(-v[:, 2], torch.norm(v[:, :2], dim=1)))
    value = ((entry - min_angle) / (full_angle - min_angle)).clamp(0.0, 1.0)
    return env.termination_manager.get_term(term_name).float() * value


##
# Curriculum
##


class reset_hoop_curriculum(ManagerTermBase):
    """Reset event that places the hoop and runs the distance curriculum.

    Each new episode gets either a *frontier* distance (``frontier ± spread``, with probability
    ``frontier_prob``) or a *review* distance drawn uniformly from ``[start_distance, frontier]``, so shots that
    were already learned are not forgotten. The frontier moves back by ``step`` once the make rate of frontier
    episodes (exponential moving average) exceeds ``promote_at``.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        p = cfg.params
        self.frontier = p.get("initial_frontier") or p["start_distance"]
        self.frontier_make_rate = 0.0
        self.frontier_prob = p.get("frontier_prob", 0.5)
        self.spread = p.get("spread", 0.25)
        self._is_frontier = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor,
        start_distance: float,
        max_distance: float,
        step: float,
        promote_at: float,
        spread: float = 0.25,
        frontier_prob: float = 0.5,
        lateral: float = 0.3,
        initial_frontier: float | None = None,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("hoop"),
        term_name: str = "made_shot",
    ):
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        # 1) update the curriculum from the episodes that just ended
        if env.common_step_counter > 0:
            mask = self._is_frontier[env_ids]
            if mask.any():
                made = env.termination_manager.get_term(term_name)[env_ids][mask].float().mean().item()
                self.frontier_make_rate = 0.98 * self.frontier_make_rate + 0.02 * made
                if self.frontier_make_rate > promote_at and self.frontier < max_distance:
                    self.frontier = min(self.frontier + step, max_distance)
                    self.frontier_make_rate = 0.0
        # 2) sample new hoop positions
        n = len(env_ids)
        dev = env.device
        frontier = torch.rand(n, device=dev) < self.frontier_prob
        d_front = self.frontier + (torch.rand(n, device=dev) * 2 - 1) * self.spread
        d_review = start_distance + torch.rand(n, device=dev) * max(self.frontier - start_distance, 0.0)
        dist = torch.where(frontier, d_front, d_review).clamp(min=start_distance - self.spread)
        y = (torch.rand(n, device=dev) * 2 - 1) * lateral
        hoop = env.scene[asset_cfg.name]
        pose = hoop.data.default_root_state[env_ids, :7].clone()
        pose[:, 0] = env.scene.env_origins[env_ids, 0] + dist
        pose[:, 1] = env.scene.env_origins[env_ids, 1] + y
        pose[:, 2] = env.scene.env_origins[env_ids, 2] + RIM_HEIGHT
        hoop.write_root_pose_to_sim(pose, env_ids=env_ids)
        hoop.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)
        self._is_frontier[env_ids] = frontier


def hoop_curriculum_state(env: ManagerBasedRLEnv, env_ids, event_name: str = "reset_hoop") -> dict[str, float]:
    """Curriculum-manager hook that only reports the state of :class:`reset_hoop_curriculum` for logging."""
    term = env.event_manager.get_term_cfg(event_name).func
    return {"frontier": term.frontier, "frontier_make_rate": term.frontier_make_rate}


class shot_distance_curriculum(ManagerTermBase):
    """Move the hoop back once the policy makes enough shots at the current distance.

    Tracks an exponential moving average of the make rate over finished episodes. When it exceeds
    ``promote_at``, the nominal rim distance grows by ``step`` (up to ``max_distance``). The hoop reset event is
    updated in place, so new episodes spawn the hoop at the new distance.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        p = cfg.params
        self.distance = p["start_distance"]
        self.make_rate = 0.0
        self._apply(env)

    def _apply(self, env: ManagerBasedRLEnv):
        p = self.cfg.params
        event = env.event_manager.get_term_cfg(p.get("event_name", "reset_hoop"))
        base_x = env.scene[p.get("hoop_name", "hoop")].cfg.init_state.pos[0]
        half = p["spread"]
        event.params["pose_range"]["x"] = (self.distance - base_x - half, self.distance - base_x + half)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        start_distance: float,
        max_distance: float,
        step: float,
        promote_at: float,
        spread: float,
        event_name: str = "reset_hoop",
        hoop_name: str = "hoop",
        term_name: str = "made_shot",
    ) -> dict[str, float]:
        if len(env_ids) > 0 and env.common_step_counter > 0:
            made = env.termination_manager.get_term(term_name)[env_ids].float().mean().item()
            self.make_rate = 0.98 * self.make_rate + 0.02 * made
            if self.make_rate > promote_at and self.distance < max_distance:
                self.distance = min(self.distance + step, max_distance)
                self.make_rate = 0.0
                self._apply(env)
        return {"distance": self.distance, "make_rate_ema": self.make_rate}
