"""Motion-prior rewards from retargeted human shots, as self-contained Isaac Lab reward terms.

Features per frame: the 14 arm joint angles plus elbow / wrist / knuckle positions (torso frame), see
``scripts/retarget_cmu_to_g1.py``.

* :class:`amp_style_reward` - Adversarial Motion Prior (Peng et al., 2021). A discriminator D sees transitions
  (s_t, s_t+1) and is trained, least-squares GAN style, to output +1 on reference motion and -1 on the policy.
  Style reward ``max(0, 1 - 0.25 * (D - 1)^2)``. D trains inside the reward term every ``update_every`` env
  steps, so RSL-RL needs no changes; rollouts run under ``torch.inference_mode`` and training escapes it.
* :class:`pose_match_reward` - dense, time-free shaping toward the reference pose manifold:
  ``exp(-min_k ||f(s) - f_ref_k||^2 / dim / scale)`` over all reference frames.

First AMP attempt (148 transitions, 512-256 D, 8 updates per step block) saturated: D scored every policy
transition at -1, the style reward went flat at 0 and gave no direction. Hence the smaller D, input noise,
stronger gradient penalty, fewer updates, 100 Hz references (2x transitions) and the pose-match shaping term.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from isaaclab.managers import ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class _Reference:
    """Reference frames and (s_t, s_t+dt) transitions at the env's policy rate."""

    def __init__(self, env: ManagerBasedRLEnv, files: list[str], robot_name: str = "robot"):
        refs = [np.load(f) for f in files]
        robot = env.scene[robot_name]
        self.joint_ids = [robot.joint_names.index(str(n)) for n in refs[0]["joint_names"]]
        self.key_ids = [robot.body_names.index(str(n)) for n in refs[0]["key_body_names"]]
        frames, trans = [], []
        for r in refs:
            f = np.concatenate([r["q"], r["key_pos"].reshape(len(r["q"]), -1)], axis=1)
            stride = max(1, int(round(float(r["fps"]) * env.step_dt)))
            frames.append(f)
            trans.append(np.concatenate([f[:-stride], f[stride:]], axis=1))
        dev = env.device
        self.frames = torch.tensor(np.concatenate(frames), dtype=torch.float32, device=dev)
        trans = torch.tensor(np.concatenate(trans), dtype=torch.float32, device=dev)
        self.f_mean = self.frames.mean(0)
        self.f_std = self.frames.std(0).clamp(min=1e-3)
        self.t_mean = trans.mean(0)
        self.t_std = trans.std(0).clamp(min=1e-3)
        self.trans = (trans - self.t_mean) / self.t_std
        self.frames_n = (self.frames - self.f_mean) / self.f_std
        self.dim = self.frames.shape[1]
        self.robot_name = robot_name

    def features(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        robot = env.scene[self.robot_name]
        q = robot.data.joint_pos[:, self.joint_ids]
        key = robot.data.body_pos_w[:, self.key_ids] - env.scene.env_origins[:, None, :]
        return torch.cat([q, key.reshape(env.num_envs, -1)], dim=1)


class pose_match_reward(ManagerTermBase):
    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.ref = _Reference(env, cfg.params["reference_files"])

    def __call__(self, env: ManagerBasedRLEnv, reference_files: list[str], scale: float = 1.0) -> torch.Tensor:
        f = (self.ref.features(env) - self.ref.f_mean) / self.ref.f_std
        d2 = torch.cdist(f, self.ref.frames_n).pow(2).min(dim=1).values / self.ref.dim
        return torch.exp(-d2 / scale)


class amp_style_reward(ManagerTermBase):
    def __init__(self, cfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        p = cfg.params
        dev = env.device
        self.ref = _Reference(env, p["reference_files"])
        n_in = self.ref.trans.shape[1]
        hidden = p.get("hidden", (256, 128))
        self.disc = nn.Sequential(
            nn.Linear(n_in, hidden[0]), nn.ReLU(), nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Linear(hidden[1], 1)
        ).to(dev)
        self.opt = torch.optim.Adam(self.disc.parameters(), lr=p.get("lr", 1e-4))
        cap = p.get("buffer_size", 200_000)
        self._buf = torch.zeros(cap, n_in, device=dev)
        self._buf_n = 0
        self._buf_ptr = 0
        self._prev = torch.zeros(env.num_envs, self.ref.dim, device=dev)
        self._prev_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
        self._calls = 0
        self._updates = 0
        self.stats = {"d_ref": 0.0, "d_policy": 0.0, "loss": 0.0}
        print(f"[AMP] {len(self.ref.trans)} reference transitions, {len(self.ref.frames)} frames, "
              f"feature dim {self.ref.dim}")

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._prev_valid[env_ids] = False  # never pair states across an episode reset

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        reference_files: list[str],
        update_every: int = 32,
        num_updates: int = 2,
        batch_size: int = 4096,
        grad_penalty: float = 10.0,
        input_noise: float = 0.1,
        lr: float = 1e-4,
        hidden: tuple[int, int] = (256, 128),
        buffer_size: int = 200_000,
    ) -> torch.Tensor:
        cur = self.ref.features(env)
        trans = (torch.cat([self._prev, cur], dim=1) - self.ref.t_mean) / self.ref.t_std
        valid = self._prev_valid.clone()
        with torch.no_grad():
            d = self.disc(trans).squeeze(-1)
        reward = torch.clamp(1.0 - 0.25 * (d - 1.0) ** 2, min=0.0) * valid.float()

        new = trans[valid]
        n = new.shape[0]
        if n > 0:
            cap = self._buf.shape[0]
            idx = (self._buf_ptr + torch.arange(n, device=new.device)) % cap
            self._buf[idx] = new
            self._buf_ptr = (self._buf_ptr + n) % cap
            self._buf_n = min(self._buf_n + n, cap)
        self._prev[:] = cur
        self._prev_valid[:] = True

        self._calls += 1
        if self._calls % update_every == 0 and self._buf_n >= batch_size:
            self._train(num_updates, batch_size, grad_penalty, input_noise)
        return reward

    def _train(self, num_updates: int, batch_size: int, grad_penalty: float, input_noise: float):
        with torch.inference_mode(False), torch.enable_grad():
            for _ in range(num_updates):
                pi = self._buf[torch.randint(0, self._buf_n, (batch_size,), device=self._buf.device)].clone()
                ref = self.ref.trans[torch.randint(0, len(self.ref.trans), (batch_size,), device=pi.device)].clone()
                pi = pi + input_noise * torch.randn_like(pi)
                ref = (ref + input_noise * torch.randn_like(ref)).requires_grad_(True)
                d_ref = self.disc(ref)
                d_pi = self.disc(pi)
                loss = ((d_ref - 1.0) ** 2).mean() + ((d_pi + 1.0) ** 2).mean()
                grad = torch.autograd.grad(d_ref.sum(), ref, create_graph=True)[0]
                loss = loss + grad_penalty * grad.pow(2).sum(dim=1).mean()
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            self.stats = {"d_ref": d_ref.mean().item(), "d_policy": d_pi.mean().item(), "loss": loss.item()}
        self._updates += 1
        if self._updates % 100 == 0:
            print(f"[AMP] update {self._updates}: D(ref)={self.stats['d_ref']:+.2f}  "
                  f"D(policy)={self.stats['d_policy']:+.2f}  loss={self.stats['loss']:.3f}")


def amp_stats(env: ManagerBasedRLEnv, env_ids, reward_name: str = "amp_style") -> dict[str, float]:
    """Curriculum-manager hook used only to log the discriminator's outputs."""
    return dict(env.reward_manager.get_term_cfg(reward_name).func.stats)
