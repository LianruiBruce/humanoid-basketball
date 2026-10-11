"""MDP terms for the robot-arm basketball task: release action, observations, rewards, terminations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import *  # noqa: F401, F403  (re-export the standard terms)
from isaaclab.envs.mdp.actions.actions_cfg import BinaryJointPositionActionCfg
from isaaclab.envs.mdp.actions.binary_joint_actions import BinaryJointPositionAction
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

RIM_RADIUS = 0.12
BALL_RADIUS = 0.025


##
# Actions
##


class ReleaseAction(BinaryJointPositionAction):
    """Gripper that holds the ball until the policy releases it.

    - Forced closed for the first ``min_hold_time`` seconds, so the policy has to wind up before it can let go.
    - Once opened it stays open for the rest of the episode (one shot per episode, no re-grasping).
    """

    cfg: ReleaseActionCfg

    def __init__(self, cfg: ReleaseActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._released = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def released(self) -> torch.Tensor:
        return self._released

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        t = self._env.episode_length_buf * self._env.step_dt
        want_open = (actions[:, 0] > 0) & (t >= self.cfg.min_hold_time)
        self._released |= want_open
        self._processed_actions = torch.where(
            self._released.unsqueeze(1), self._open_command, self._close_command
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._released[env_ids] = False


@configclass
class ReleaseActionCfg(BinaryJointPositionActionCfg):
    class_type: type = ReleaseAction
    min_hold_time: float = 0.3


##
# Observations
##


def position_in_robot_frame(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    asset = env.scene[asset_cfg.name]
    pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, asset.data.root_pos_w)
    return pos_b


def episode_time(env: ManagerBasedRLEnv) -> torch.Tensor:
    return (env.episode_length_buf * env.step_dt).unsqueeze(1)


def is_released(env: ManagerBasedRLEnv, action_name: str = "gripper_action") -> torch.Tensor:
    return env.action_manager.get_term(action_name).released.float().unsqueeze(1)


##
# Terminations
##


class ball_through_hoop(ManagerTermBase):
    """Success: the ball center crosses the rim plane downward inside the rim.

    Keeps the ball height (relative to the rim) from the previous step to detect the crossing; the horizontal
    position at the crossing is linearly interpolated between the two steps.
    """

    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._prev_rel = torch.full((env.num_envs, 3), float("nan"), device=env.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._prev_rel[env_ids] = float("nan")

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
        hoop_cfg: SceneEntityCfg = SceneEntityCfg("hoop"),
        rim_radius: float = RIM_RADIUS,
    ) -> torch.Tensor:
        rel = env.scene[ball_cfg.name].data.root_pos_w - env.scene[hoop_cfg.name].data.root_pos_w
        prev = self._prev_rel
        crossed = (prev[:, 2] > 0) & (rel[:, 2] <= 0)  # NaN compares False -> no crossing on first step
        # fraction of the step at which z hits 0
        frac = (prev[:, 2] / (prev[:, 2] - rel[:, 2]).clamp(min=1e-6)).clamp(0.0, 1.0).unsqueeze(1)
        xy_at_cross = prev[:, :2] + frac * (rel[:, :2] - prev[:, :2])
        inside = torch.norm(xy_at_cross, dim=1) < rim_radius - 0.5 * BALL_RADIUS
        self._prev_rel[:] = rel
        return crossed & inside


def ball_missed(
    env: ManagerBasedRLEnv,
    margin: float = 0.15,
    floor_height: float = -0.5,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    hoop_cfg: SceneEntityCfg = SceneEntityCfg("hoop"),
    action_name: str = "gripper_action",
) -> torch.Tensor:
    """Released ball is falling and already well below the rim, or it fell off the table."""
    ball = env.scene[ball_cfg.name].data
    hoop_z = env.scene[hoop_cfg.name].data.root_pos_w[:, 2]
    released = env.action_manager.get_term(action_name).released
    falling_below = (ball.root_lin_vel_w[:, 2] < 0) & (ball.root_pos_w[:, 2] < hoop_z - margin)
    return (released & falling_below) | (ball.root_pos_w[:, 2] < floor_height)


##
# Rewards
##


def ball_near_hoop(
    env: ManagerBasedRLEnv,
    std: float,
    aim_height: float = 0.1,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    hoop_cfg: SceneEntityCfg = SceneEntityCfg("hoop"),
    action_name: str = "gripper_action",
) -> torch.Tensor:
    """Dense shaping while the ball is in flight: closeness to a point just above the rim center."""
    ball_pos = env.scene[ball_cfg.name].data.root_pos_w
    aim = env.scene[hoop_cfg.name].data.root_pos_w.clone()
    aim[:, 2] += aim_height
    dist = torch.norm(ball_pos - aim, dim=1)
    released = env.action_manager.get_term(action_name).released
    return released.float() * torch.exp(-dist / std)


def made_shot(env: ManagerBasedRLEnv, term_name: str = "made_shot") -> torch.Tensor:
    return env.termination_manager.get_term(term_name).float()
