"""Whole-body motion tracking for the G1 (BeyondMimic-style), used to learn a human shooting motion.

:class:`MotionCommand` owns the reference clip. Each env keeps its own frame index; on reset it starts at a random
frame and the robot is placed in that reference state (reference state initialization). Rewards compare the
robot to the reference with exponential kernels: the anchor (torso) pose in the world, the other bodies relative
to the anchor after aligning headings, and body velocities.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_conjugate,
    quat_error_magnitude,
    quat_mul,
    subtract_frame_transforms,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


##
# Reference motion command
##


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        dev = self.device
        data = np.load(cfg.motion_file)
        tt = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=dev)  # noqa: E731
        m_joints = [str(n) for n in data["joint_names"]]
        m_bodies = [str(n) for n in data["body_names"]]
        j = [m_joints.index(n) for n in self.robot.joint_names]  # motion columns in robot joint order
        self.joint_pos = tt(data["joint_pos"])[:, j]
        self.joint_vel = tt(data["joint_vel"])[:, j]
        self.body_names = list(cfg.body_names)
        b = [m_bodies.index(n) for n in self.body_names]
        self.body_pos = tt(data["body_pos_w"])[:, b]
        self.body_quat = tt(data["body_quat_w"])[:, b]
        self.body_lin_vel = tt(data["body_lin_vel_w"])[:, b]
        self.body_ang_vel = tt(data["body_ang_vel_w"])[:, b]
        root = m_bodies.index(cfg.root_body)
        self.root_pos = tt(data["body_pos_w"])[:, root]
        self.root_quat = tt(data["body_quat_w"])[:, root]
        self.root_lin_vel = tt(data["body_lin_vel_w"])[:, root]
        self.root_ang_vel = tt(data["body_ang_vel_w"])[:, root]
        self.num_frames = self.joint_pos.shape[0]
        self.robot_body_ids, _ = self.robot.find_bodies(self.body_names, preserve_order=True)
        self.anchor_idx = self.body_names.index(cfg.anchor_body)
        self.robot_anchor_id = self.robot_body_ids[self.anchor_idx]
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=dev)
        for k in ("anchor_pos", "anchor_ori", "body_pos", "joint_pos"):
            self.metrics[f"error_{k}"] = torch.zeros(self.num_envs, device=dev)
        print(f"[tracking] {cfg.motion_file}: {self.num_frames} frames, {len(self.body_names)} tracked bodies")

    # --- reference at the current frame (world frame, env origin added) ---
    @property
    def command(self) -> torch.Tensor:
        t = self.time_steps
        return torch.cat([self.joint_pos[t], self.joint_vel[t]], dim=1)

    @property
    def ref_body_pos_w(self) -> torch.Tensor:
        return self.body_pos[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def ref_body_quat_w(self) -> torch.Tensor:
        return self.body_quat[self.time_steps]

    @property
    def ref_body_lin_vel_w(self) -> torch.Tensor:
        return self.body_lin_vel[self.time_steps]

    @property
    def ref_body_ang_vel_w(self) -> torch.Tensor:
        return self.body_ang_vel[self.time_steps]

    @property
    def ref_anchor_pos_w(self) -> torch.Tensor:
        return self.ref_body_pos_w[:, self.anchor_idx]

    @property
    def ref_anchor_quat_w(self) -> torch.Tensor:
        return self.ref_body_quat_w[:, self.anchor_idx]

    @property
    def ref_joint_pos(self) -> torch.Tensor:
        return self.joint_pos[self.time_steps]

    # --- robot ---
    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_body_ids]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_body_ids]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_id]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_id]

    def aligned_ref_bodies(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference bodies moved onto the robot: same heading and xy as the robot anchor, reference height."""
        delta = yaw_quat(quat_mul(self.robot_anchor_quat_w, quat_conjugate(self.ref_anchor_quat_w)))
        delta = delta[:, None, :].expand(-1, len(self.body_names), -1)
        anchor = self.ref_anchor_pos_w.clone()
        anchor[:, :2] = self.robot_anchor_pos_w[:, :2]
        rel = self.ref_body_pos_w - self.ref_anchor_pos_w[:, None, :]
        pos = anchor[:, None, :] + quat_apply(delta, rel)
        quat = quat_mul(delta, self.ref_body_quat_w)
        return pos, quat

    # --- CommandTerm hooks ---
    def _update_metrics(self):
        self.metrics["error_anchor_pos"] = torch.norm(self.ref_anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_ori"] = quat_error_magnitude(self.ref_anchor_quat_w, self.robot_anchor_quat_w)
        pos, _ = self.aligned_ref_bodies()
        self.metrics["error_body_pos"] = torch.norm(pos - self.robot_body_pos_w, dim=-1).mean(-1)
        self.metrics["error_joint_pos"] = torch.norm(self.ref_joint_pos - self.robot.data.joint_pos, dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        """Reference state initialization: start at a random frame, in the reference pose plus a little noise."""
        if len(env_ids) == 0:
            return
        n = len(env_ids)
        dev = self.device
        t0 = torch.randint(0, max(self.num_frames - self.cfg.min_frames_left, 1), (n,), device=dev)
        r = lambda lo, hi, shape: torch.empty(shape, device=dev).uniform_(lo, hi)  # noqa: E731
        root_pos = self.root_pos[t0] + self._env.scene.env_origins[env_ids]
        root_pos[:, :2] += r(-self.cfg.pose_noise, self.cfg.pose_noise, (n, 2))
        root_pos[:, 2] += r(0.0, 0.02, (n,))
        root_vel = torch.cat([self.root_lin_vel[t0], self.root_ang_vel[t0]], dim=1)
        root_vel += r(-self.cfg.vel_noise, self.cfg.vel_noise, (n, 6))
        jp = self.joint_pos[t0] + r(-self.cfg.joint_noise, self.cfg.joint_noise, (n, self.joint_pos.shape[1]))
        limits = self.robot.data.soft_joint_pos_limits[env_ids]
        jp = torch.clamp(jp, limits[..., 0], limits[..., 1])
        self.robot.write_root_pose_to_sim(torch.cat([root_pos, self.root_quat[t0]], dim=1), env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(root_vel, env_ids=env_ids)
        self.robot.write_joint_state_to_sim(jp, self.joint_vel[t0], env_ids=env_ids)
        self.time_steps[env_ids] = torch.clamp(t0 + 1, max=self.num_frames - 1)

    def _update_command(self):
        self.time_steps = torch.clamp(self.time_steps + 1, max=self.num_frames - 1)


@configclass
class MotionCommandCfg(CommandTermCfg):
    class_type: type = MotionCommand
    asset_name: str = "robot"
    motion_file: str = MISSING
    body_names: list[str] = MISSING
    anchor_body: str = "torso_link"
    root_body: str = "pelvis"
    min_frames_left: int = 10
    pose_noise: float = 0.05
    vel_noise: float = 0.1
    joint_noise: float = 0.05
    resampling_time_range: tuple[float, float] = (1.0e6, 1.0e6)  # only resample on reset


##
# Observations
##


def _cmd(env: ManagerBasedRLEnv, name: str) -> MotionCommand:
    return env.command_manager.get_term(name)


def motion_anchor_pos_b(env: ManagerBasedRLEnv, command_name: str = "motion") -> torch.Tensor:
    c = _cmd(env, command_name)
    pos, _ = subtract_frame_transforms(c.robot_anchor_pos_w, c.robot_anchor_quat_w, c.ref_anchor_pos_w,
                                       c.ref_anchor_quat_w)
    return pos


def motion_anchor_ori_b(env: ManagerBasedRLEnv, command_name: str = "motion") -> torch.Tensor:
    """Reference anchor orientation in the robot anchor frame, as the first two columns of the rotation matrix."""
    c = _cmd(env, command_name)
    _, quat = subtract_frame_transforms(c.robot_anchor_pos_w, c.robot_anchor_quat_w, c.ref_anchor_pos_w,
                                        c.ref_anchor_quat_w)
    return matrix_from_quat(quat)[..., :2].reshape(env.num_envs, -1)


def motion_phase(env: ManagerBasedRLEnv, command_name: str = "motion") -> torch.Tensor:
    c = _cmd(env, command_name)
    return (c.time_steps.float() / max(c.num_frames - 1, 1)).unsqueeze(1)


def robot_body_pos_b(env: ManagerBasedRLEnv, command_name: str = "motion") -> torch.Tensor:
    """Critic only: tracked bodies in the robot anchor frame."""
    c = _cmd(env, command_name)
    n = len(c.body_names)
    pos, _ = subtract_frame_transforms(
        c.robot_anchor_pos_w[:, None, :].expand(-1, n, -1), c.robot_anchor_quat_w[:, None, :].expand(-1, n, -1),
        c.robot_body_pos_w, c.robot_body_quat_w)
    return pos.reshape(env.num_envs, -1)


##
# Rewards
##


def anchor_pos_exp(env: ManagerBasedRLEnv, std: float, command_name: str = "motion") -> torch.Tensor:
    c = _cmd(env, command_name)
    err = torch.sum((c.ref_anchor_pos_w - c.robot_anchor_pos_w) ** 2, dim=-1)
    return torch.exp(-err / std**2)


def anchor_ori_exp(env: ManagerBasedRLEnv, std: float, command_name: str = "motion") -> torch.Tensor:
    c = _cmd(env, command_name)
    err = quat_error_magnitude(c.ref_anchor_quat_w, c.robot_anchor_quat_w) ** 2
    return torch.exp(-err / std**2)


def body_pos_exp(env: ManagerBasedRLEnv, std: float, command_name: str = "motion") -> torch.Tensor:
    c = _cmd(env, command_name)
    pos, _ = c.aligned_ref_bodies()
    err = torch.sum((pos - c.robot_body_pos_w) ** 2, dim=-1).mean(-1)
    return torch.exp(-err / std**2)


def body_ori_exp(env: ManagerBasedRLEnv, std: float, command_name: str = "motion") -> torch.Tensor:
    c = _cmd(env, command_name)
    _, quat = c.aligned_ref_bodies()
    err = (quat_error_magnitude(quat, c.robot_body_quat_w) ** 2).mean(-1)
    return torch.exp(-err / std**2)


def body_lin_vel_exp(env: ManagerBasedRLEnv, std: float, command_name: str = "motion") -> torch.Tensor:
    c = _cmd(env, command_name)
    vel = c.robot.data.body_lin_vel_w[:, c.robot_body_ids]
    err = torch.sum((c.ref_body_lin_vel_w - vel) ** 2, dim=-1).mean(-1)
    return torch.exp(-err / std**2)


def body_ang_vel_exp(env: ManagerBasedRLEnv, std: float, command_name: str = "motion") -> torch.Tensor:
    c = _cmd(env, command_name)
    vel = c.robot.data.body_ang_vel_w[:, c.robot_body_ids]
    err = torch.sum((c.ref_body_ang_vel_w - vel) ** 2, dim=-1).mean(-1)
    return torch.exp(-err / std**2)


##
# Terminations
##


def anchor_height_error(env: ManagerBasedRLEnv, threshold: float, command_name: str = "motion") -> torch.Tensor:
    c = _cmd(env, command_name)
    return torch.abs(c.ref_anchor_pos_w[:, 2] - c.robot_anchor_pos_w[:, 2]) > threshold


def anchor_tilt_error(env: ManagerBasedRLEnv, threshold: float, command_name: str = "motion") -> torch.Tensor:
    """Gravity seen from the reference vs. the robot anchor differs by more than ``threshold`` (z component)."""
    c = _cmd(env, command_name)
    g = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(env.num_envs, 3)
    g_ref = quat_apply(quat_conjugate(c.ref_anchor_quat_w), g)
    g_rob = quat_apply(quat_conjugate(c.robot_anchor_quat_w), g)
    return torch.abs(g_ref[:, 2] - g_rob[:, 2]) > threshold


def end_effector_height_error(env: ManagerBasedRLEnv, threshold: float, body_names: list[str],
                              command_name: str = "motion") -> torch.Tensor:
    c = _cmd(env, command_name)
    ids = [c.body_names.index(n) for n in body_names]
    pos, _ = c.aligned_ref_bodies()
    err = torch.abs(pos[:, ids, 2] - c.robot_body_pos_w[:, ids, 2])
    return torch.any(err > threshold, dim=-1)


def motion_finished(env: ManagerBasedRLEnv, command_name: str = "motion") -> torch.Tensor:
    c = _cmd(env, command_name)
    return c.time_steps >= c.num_frames - 1
