"""Retarget a CMU mocap clip onto the whole G1 body (legs, waist, arms) for motion tracking.

* Root: the human pelvis trajectory, translation scaled by the G1/human leg-length ratio, heading rotated so the
  shooter faces +X (the hoop direction in the basketball task).
* Joints: damped-least-squares IK in the pelvis frame. Knee/ankle/elbow/wrist/knuckle targets are rebuilt from the
  human segment directions with the G1's own segment lengths; feet and torso also match the human orientation.
  Every frame is its own env, so all frames are solved in parallel.
* Ground: each frame the lowest G1 sole is placed at the (scaled) height of the lowest human foot above the human
  floor, so feet stay planted while standing and leave the ground during a jump.

    python scripts/retarget_cmu_wholebody.py --headless --amc data/mocap/cmu_124/124_05.amc --start 2.6 --end 4.6 \
        --out data/mocap/wholebody/g1_124_05.npz
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--amc", required=True)
parser.add_argument("--asf", default=None)
parser.add_argument("--start", type=float, required=True, help="seconds")
parser.add_argument("--end", type=float, required=True, help="seconds")
parser.add_argument("--fps", type=float, default=50.0)
parser.add_argument("--iters", type=int, default=120)
parser.add_argument("--out", required=True)
parser.add_argument("--ball_hold", action="store_true",
                    help="before the release, keep the hands at least --hold_gap apart (room for the ball)")
parser.add_argument("--hold_gap", type=float, default=0.30, help="min wrist / knuckle separation while holding (m)")
parser.add_argument("--smooth", type=float, default=0.3, help="weight of the temporal joint-smoothness term")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import ArticulationCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.math import axis_angle_from_quat, matrix_from_quat, quat_conjugate, quat_from_matrix, quat_mul  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.join(ROOT, "source"))
from basketball.g1_shoot_env_cfg import ROBOT_CFG  # noqa: E402
from basketball.mocap.cmu import load_positions  # noqa: E402

SOLE_BELOW_ANKLE = 0.036  # ankle_roll_link origin is ~3.6 cm above the sole
BODY_JOINTS = [
    f"{s}_{j}_joint" for s in ("left", "right")
    for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
] + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"] + [
    f"{s}_{j}_joint" for s in ("left", "right")
    for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
]


def _unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True).clip(1e-9)


def _frame(fwd_hint, left, up_hint=None):
    """Rotation matrices (T,3,3) with columns [x fwd, y left, z up] from a left vector and an up (or fwd) hint."""
    left = _unit(left)
    if up_hint is not None:
        fwd = _unit(np.cross(left, up_hint))
    else:
        fwd = _unit(fwd_hint - np.sum(fwd_hint * left, -1, keepdims=True) * left)
    up = np.cross(fwd, left)
    return np.stack([fwd, left, up], axis=-1)


def human_motion(names, pos):
    """Pelvis/torso/foot frames and segment directions, all in the human pelvis frame."""
    i = {n: k for k, n in enumerate(names)}
    P = lambda n: pos[:, i[n]]  # noqa: E731
    pelvis = P("root")
    R_pel = _frame(None, P("lhipjoint") - P("rhipjoint"), P("lowerback") - pelvis)
    R_tor = _frame(None, P("lclavicle") - P("rclavicle"), P("thorax") - P("lowerback"))
    to_local = lambda v: np.einsum("tji,tj->ti", R_pel, v)  # noqa: E731  (R^T v)
    out = {"R_pel": R_pel, "R_torso_local": np.einsum("tji,tjk->tik", R_pel, R_tor), "pelvis": pelvis}
    for side, h in (("left", "l"), ("right", "r")):
        hip, knee, ankle, toe = P(f"{h}hipjoint"), P(f"{h}femur"), P(f"{h}tibia"), P(f"{h}toes")
        out[f"{side}_leg_dirs"] = [_unit(to_local(knee - hip)), _unit(to_local(ankle - knee))]
        # ankle->toe points ~20-30 deg down even when standing flat; remove the clip's median pitch so a flat human
        # foot maps to a flat G1 foot
        f = toe - ankle
        horiz = _unit(f * np.array([1.0, 1.0, 0.0]))
        pitch = np.arctan2(-f[:, 2], np.linalg.norm(f[:, :2], axis=-1))
        pitch = pitch - np.median(pitch)
        f_flat = np.cos(pitch)[:, None] * horiz - np.sin(pitch)[:, None] * np.array([0.0, 0.0, 1.0])
        R_foot = _frame(f_flat, P("lhipjoint") - P("rhipjoint"))
        out[f"{side}_foot_local"] = np.einsum("tji,tjk->tik", R_pel, R_foot)
        sh, el, wr, hd = P(f"{h}clavicle"), P(f"{h}humerus"), P(f"{h}radius"), P(f"{h}hand")
        out[f"{side}_arm_dirs"] = [_unit(to_local(el - sh)), _unit(to_local(wr - el)), _unit(to_local(hd - wr))]
    feet = np.stack([P("lfoot"), P("rfoot"), P("ltoes"), P("rtoes"), P("ltibia"), P("rtibia")], 1)
    floor = feet[..., 2].min()
    out["foot_height"] = feet[..., 2].min(1) - floor  # lowest foot above the human floor
    for side, h in (("left", "l"), ("right", "r")):  # per-foot height above the floor and ankle ground track
        out[f"{side}_foot_h"] = np.stack([P(f"{h}foot"), P(f"{h}toes"), P(f"{h}tibia")], 1)[..., 2].min(1) - floor
        out[f"{side}_ankle_xy"] = P(f"{h}tibia")[:, :2]
    out["leg_len"] = np.mean([np.linalg.norm(P(f"{h}femur") - P(f"{h}hipjoint"), axis=-1).mean()
                              + np.linalg.norm(P(f"{h}tibia") - P(f"{h}femur"), axis=-1).mean() for h in "lr"])
    return out


robot_cfg = ROBOT_CFG.replace(init_state=ROBOT_CFG.init_state.replace(pos=(0.0, 0.0, 1.0)))


@configclass
class SceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = robot_cfg


def main():
    asf = args.asf or os.path.join(os.path.dirname(args.amc), "124.asf")
    names, pos = load_positions(asf, args.amc)
    t_src = np.arange(len(pos)) / 120.0
    t_out = np.arange(args.start, args.end, 1.0 / args.fps)
    pos = np.stack([np.stack([np.interp(t_out, t_src, pos[:, j, k]) for k in range(3)], -1)
                    for j in range(pos.shape[1])], 1)
    # face +X: rotate the clip about Z so the pelvis heading in the middle frame points along +X
    i = {n: k for k, n in enumerate(names)}
    mid = len(pos) // 2
    left = pos[mid, i["lhipjoint"]] - pos[mid, i["rhipjoint"]]
    fwd = np.cross(left, [0.0, 0.0, 1.0])
    yaw = np.arctan2(fwd[1], fwd[0])
    c, s = np.cos(-yaw), np.sin(-yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    pos = pos - np.array([pos[0, i["root"], 0], pos[0, i["root"], 1], 0.0])
    pos = pos @ Rz.T
    hm = human_motion(names, pos)
    T = len(t_out)
    # release frame: the right (shooting) wrist reaches its highest point relative to the pelvis
    rw = pos[:, i["rwrist"], 2] - pos[:, i["root"], 2]
    release_frame = int(np.argmax(rw))
    print(f"[wb] release frame {release_frame} (t={t_out[release_frame]:.2f}s)")
    print(f"[wb] {T} frames at {args.fps:.0f} Hz from {os.path.basename(args.amc)} [{args.start}, {args.end}] s")

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, gravity=(0.0, 0.0, 0.0)))
    scene = InteractiveScene(SceneCfg(num_envs=T, env_spacing=2.0))
    sim.reset()
    robot = scene["robot"]
    dev = robot.device
    bn, jn = robot.body_names, robot.joint_names
    scene.update(0.0)
    pel0 = robot.data.body_pos_w[:, bn.index("pelvis")].clone()

    # G1 segment lengths (zero pose) and the leg-length scale
    b = lambda n: bn.index(n)  # noqa: E731
    p0 = robot.data.body_pos_w[0]
    seg = lambda a, c: torch.norm(p0[b(c)] - p0[b(a)]).item()  # noqa: E731
    g1_leg = seg("left_hip_roll_link", "left_knee_link") + seg("left_knee_link", "left_ankle_roll_link")
    scale = g1_leg / hm["leg_len"]
    print(f"[wb] leg length human {hm['leg_len']:.3f} m, G1 {g1_leg:.3f} m -> scale {scale:.3f}")

    chains = {}
    for side, H in (("left", "L"), ("right", "R")):
        chains[f"{side}_leg"] = [b(f"{side}_hip_roll_link"), b(f"{side}_knee_link"), b(f"{side}_ankle_roll_link")]
        chains[f"{side}_arm"] = [b(f"{side}_shoulder_roll_link"), b(f"{side}_elbow_link"),
                                 b(f"{side}_wrist_yaw_link"), b(f"{H}_middle_proximal")]
    seglen = {k: [torch.norm(p0[v[j + 1]] - p0[v[j]]).item() for j in range(len(v) - 1)] for k, v in chains.items()}

    tt = lambda x: torch.tensor(x, dtype=torch.float32, device=dev)  # noqa: E731
    dirs = {k: [tt(d) for d in hm[f"{k}_dirs"]] for k in chains}
    # 1 while holding, linear fade to 0 over the 10 frames after the release (an instant switch made the hands snap
    # together at the release: joint speeds jumped from 21 to 41 rad/s)
    frames = torch.arange(T, device=dev, dtype=torch.float32)
    hold_weight = (1.0 - (frames - release_frame) / 10.0).clamp(0.0, 1.0)
    rot_targets = {  # body -> desired orientation in the pelvis frame (as quaternions)
        b("left_ankle_roll_link"): quat_from_matrix(tt(hm["left_foot_local"])),
        b("right_ankle_roll_link"): quat_from_matrix(tt(hm["right_foot_local"])),
        b("torso_link"): quat_from_matrix(tt(hm["R_torso_local"])),
    }
    rot_w = {b("left_ankle_roll_link"): 0.3, b("right_ankle_roll_link"): 0.3, b("torso_link"): 0.5}
    pos_w = [0.5, 1.0, 1.0]  # knee/elbow, ankle/wrist, knuckle

    jids = [jn.index(n) for n in BODY_JOINTS]
    lo, hi = robot.data.soft_joint_pos_limits[..., 0], robot.data.soft_joint_pos_limits[..., 1]
    lam = 0.05
    ankle_ids = {"left_leg": b("left_ankle_roll_link"), "right_leg": b("right_ankle_roll_link")}

    def solve(q, iters, locks=None, tag=""):
        """DLS IK in the pelvis frame. ``locks``: {leg: (targets (T,3) in IK-env coords, mask (T,))} pins ankles."""
        for it in range(iters):
            robot.write_joint_state_to_sim(q, torch.zeros_like(q))
            sim.forward()
            scene.update(0.0)
            jac = robot.root_physx_view.get_jacobians()[..., jids]  # (T, bodies-1, 6, 29), fixed base
            bp = robot.data.body_pos_w
            rows_J, rows_e, pos_err, rot_err = [], [], [], []
            # chain targets from the human segment directions
            targets = {}
            for k, chain in chains.items():
                p = bp[:, chain[0]]
                targets[k] = []
                for j in range(len(chain) - 1):
                    p = p + dirs[k][j] * seglen[k][j]
                    if locks is not None and k in locks and j == 1:
                        tgt, mask = locks[k]
                        p = torch.where(mask[:, None], tgt, p)
                    targets[k].append(p)
            # ball-aware hands: before the release, keep wrists/knuckles at least `hold_gap` apart so a size-7 ball
            # fits between the palms (the mocap subject held their hands closer than a real ball allows)
            if args.ball_hold:
                gap = args.hold_gap * hold_weight[:, None]  # fades out over 10 frames after the release
                for j in (1, 2):
                    pl, pr = targets["left_arm"][j], targets["right_arm"][j]
                    mid, d = (pl + pr) / 2, pl - pr
                    dist = torch.norm(d, dim=-1, keepdim=True).clamp(min=1e-6)
                    widen = dist < gap
                    half = d / dist * gap / 2
                    targets["left_arm"][j] = torch.where(widen, mid + half, pl)
                    targets["right_arm"][j] = torch.where(widen, mid - half, pr)
            for k, chain in chains.items():
                for j in range(len(chain) - 1):
                    w = torch.full((T, 1), pos_w[min(j, 2)] if "arm" in k else pos_w[min(j, 1)], device=dev)
                    if locks is not None and k in locks and j == 1:
                        w = torch.where(locks[k][1][:, None], torch.full_like(w, 3.0), w)
                    e = targets[k][j] - bp[:, chain[j + 1]]
                    rows_J.append(w[:, :, None] * jac[:, chain[j + 1] - 1, :3])
                    rows_e.append(w * e)
                    pos_err.append(torch.norm(e, dim=-1))
            for body, q_des in rot_targets.items():
                e = axis_angle_from_quat(quat_mul(q_des, quat_conjugate(robot.data.body_quat_w[:, body])))
                rows_J.append(rot_w[body] * jac[:, body - 1, 3:6])
                rows_e.append(rot_w[body] * e)
                rot_err.append(torch.norm(e, dim=-1))
            # temporal smoothness: pull each frame toward the mean of its neighbours. Frames are solved in parallel,
            # so without this, neighbouring frames can land on different wrist IK branches (a 2 rad flip in one frame)
            if args.smooth > 0:
                qb = q[:, jids]
                q_nb = torch.cat([qb[:1], (qb[:-2] + qb[2:]) / 2, qb[-1:]], dim=0)
                rows_J.append(args.smooth * torch.eye(len(jids), device=dev).expand(T, -1, -1))
                rows_e.append(args.smooth * (q_nb - qb))
            J = torch.cat(rows_J, dim=1)
            e = torch.cat(rows_e, dim=1).unsqueeze(-1)
            JT = J.transpose(1, 2)
            dq = JT @ torch.linalg.solve(J @ JT + lam**2 * torch.eye(J.shape[1], device=dev), e)
            q[:, jids] += dq.squeeze(-1).clamp(-0.2, 0.2)
            q = torch.max(torch.min(q, hi), lo)
            if it % 30 == 0 or it == iters - 1:
                print(f"[wb] {tag}iter {it:3d}  position error {torch.stack(pos_err).mean() * 100:.1f} cm  "
                      f"orientation error {torch.rad2deg(torch.stack(rot_err).mean()):.1f} deg")
        return q

    def body_local(q):
        robot.write_joint_state_to_sim(q, torch.zeros_like(q))
        sim.forward()
        scene.update(0.0)
        return (robot.data.body_pos_w - pel0[:, None, :]).cpu().numpy()  # bodies relative to the pelvis

    # pass 1: free IK, then root trajectory + ground contact
    q = solve(robot.data.default_joint_pos.clone(), args.iters, tag="pass 1 ")
    local = body_local(q)
    R_pel = hm["R_pel"]
    root_xy = scale * hm["pelvis"][:, :2]
    soles = [b("left_ankle_roll_link"), b("right_ankle_roll_link")]
    sole_rel_z = np.einsum("tij,tkj->tki", R_pel, local[:, soles])[..., 2] - SOLE_BELOW_ANKLE  # rotated, root-relative
    root_z = scale * hm["foot_height"] - sole_rel_z.min(1)
    root_pos = np.concatenate([root_xy, root_z[:, None]], 1)
    root_quat = quat_from_matrix(tt(R_pel)).cpu().numpy()  # wxyz
    world = lambda loc: root_pos[:, None, :] + np.einsum("tij,tkj->tki", R_pel, loc)  # noqa: E731

    # pass 2: foot-contact cleanup. In each stance segment (sole on the ground, slow) pin the ankle to one spot on
    # the floor and re-solve, so the policy is never asked to reproduce a sliding foot.
    body_pos = world(local)
    locks = {}
    for leg, bid in ankle_ids.items():
        a = body_pos[:, bid]
        # stance comes from the human foot (on the floor and not moving), not from the G1 pass-1 result, where only
        # the lowest foot is guaranteed to touch the ground
        side = leg.split("_")[0]
        h_speed = np.r_[0.0, np.linalg.norm(np.diff(hm[f"{side}_ankle_xy"], axis=0), axis=1) * args.fps]
        stance = (hm[f"{side}_foot_h"] < 0.04) & (h_speed < 0.5)
        tgt_w = a.copy()
        t = 0
        while t < T:
            if not stance[t]:
                t += 1
                continue
            s = t
            while t < T and stance[t]:
                t += 1
            if t - s >= 3:
                tgt_w[s:t, :2] = np.median(a[s:t, :2], axis=0)
                tgt_w[s:t, 2] = SOLE_BELOW_ANKLE
            else:
                stance[s:t] = False
        tgt_local = np.einsum("tji,tj->ti", R_pel, tgt_w - root_pos)  # R^T (target - root)
        locks[leg] = (tt(tgt_local) + pel0, torch.tensor(stance, device=dev))
        print(f"[wb] {leg}: {int(stance.sum())}/{T} stance frames pinned")
    q = solve(q, args.iters // 2, locks=locks, tag="pass 2 ")
    body_pos = world(body_local(q))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, fps=args.fps, joint_names=np.array(BODY_JOINTS), joint_pos=q[:, jids].cpu().numpy(),
             root_pos=root_pos, root_quat=root_quat, body_names=np.array(bn), body_pos=body_pos,
             source=os.path.basename(args.amc), window=np.array([args.start, args.end]), scale=scale,
             release_frame=release_frame, ball_hold=args.ball_hold)
    print(f"[wb] pelvis height {root_z.min():.2f}-{root_z.max():.2f} m, "
          f"root travel {np.linalg.norm(root_xy[-1] - root_xy[0]):.2f} m")
    print(f"[wb] saved {args.out}")


main()
os._exit(0)
