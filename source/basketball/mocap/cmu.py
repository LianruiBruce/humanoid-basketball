"""Minimal ASF/AMC (CMU motion capture) reader with forward kinematics.

Joint positions come out in meters in a Z-up frame. CMU data is Y-up with lengths in ASF units, where
1 unit = (1 / 0.45) inch.
"""

from __future__ import annotations

import numpy as np

UNIT_TO_M = (1.0 / 0.45) * 0.0254


def _rot(axis: str, deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _euler_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    """ASF 'XYZ' order: rotate about x, then y, then z (fixed axes) -> R = Rz Ry Rx."""
    return _rot("z", rz) @ _rot("y", ry) @ _rot("x", rx)


class Bone:
    def __init__(self, name: str):
        self.name = name
        self.direction = np.zeros(3)
        self.length = 0.0
        self.axis = np.eye(3)
        self.dofs: list[str] = []
        self.children: list[str] = []


def read_asf(path: str) -> dict[str, Bone]:
    bones = {"root": Bone("root")}
    lines = [ln.strip() for ln in open(path)]
    i = 0
    section = None
    cur = None
    while i < len(lines):
        ln = lines[i]
        if ln.startswith(":"):
            section = ln.split()[0]
        elif section == ":bonedata":
            tok = ln.split()
            if tok[0] == "begin":
                cur = None
            elif tok[0] == "name":
                cur = Bone(tok[1])
                bones[cur.name] = cur
            elif tok[0] == "direction":
                cur.direction = np.array([float(x) for x in tok[1:4]])
            elif tok[0] == "length":
                cur.length = float(tok[1])
            elif tok[0] == "axis":
                cur.axis = _euler_xyz(*[float(x) for x in tok[1:4]])
            elif tok[0] == "dof":
                cur.dofs = tok[1:]
        elif section == ":hierarchy":
            tok = ln.split()
            if tok and tok[0] not in ("begin", "end"):
                bones[tok[0]].children += tok[1:]
        i += 1
    return bones


def read_amc(path: str) -> list[dict[str, list[float]]]:
    frames, cur = [], None
    for ln in open(path):
        ln = ln.strip()
        if not ln or ln.startswith("#") or ln.startswith(":"):
            continue
        tok = ln.split()
        if len(tok) == 1 and tok[0].isdigit():
            cur = {}
            frames.append(cur)
        elif cur is not None:
            cur[tok[0]] = [float(x) for x in tok[1:]]
    return frames


def forward_kinematics(bones: dict[str, Bone], frame: dict[str, list[float]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return {bone: (start_position, end_position)} in ASF units, Y-up."""
    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    root = frame["root"]
    root_pos = np.array(root[:3])
    root_rot = _euler_xyz(*root[3:6])
    out["root"] = (root_pos, root_pos)

    def visit(name: str, parent_rot: np.ndarray, start: np.ndarray):
        b = bones[name]
        angles = {"rx": 0.0, "ry": 0.0, "rz": 0.0}
        for dof, val in zip(b.dofs, frame.get(name, [])):
            angles[dof] = val
        local = _euler_xyz(angles["rx"], angles["ry"], angles["rz"])
        rot = parent_rot @ b.axis @ local @ b.axis.T
        end = start + b.length * (rot @ b.direction)
        out[name] = (start, end)
        for c in b.children:
            visit(c, rot, end)

    for c in bones["root"].children:
        visit(c, root_rot, root_pos)
    return out


def load_positions(asf: str, amc: str) -> tuple[list[str], np.ndarray]:
    """All bone end positions, shape (frames, bones, 3), meters, Z-up (x, y, z) = (cmu_z, cmu_x, cmu_y)."""
    bones = read_asf(asf)
    frames = read_amc(amc)
    names = [n for n in bones if n != "root"]
    pos = np.zeros((len(frames), len(names) + 1, 3))
    for t, fr in enumerate(frames):
        fk = forward_kinematics(bones, fr)
        pos[t, 0] = fk["root"][1]
        for j, n in enumerate(names):
            pos[t, j + 1] = fk[n][1]
    pos = pos[..., [2, 0, 1]] * UNIT_TO_M  # Y-up -> Z-up
    return ["root"] + names, pos
