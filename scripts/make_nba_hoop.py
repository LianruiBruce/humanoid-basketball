"""Generate source/basketball/assets/nba_hoop.usda: a regulation NBA hoop plus half-court floor markings.

Origin = rim center (put it 3.05 m above the floor). The shooter stands on the -X side, the backboard is at +X.
Rim, backboard, bracket and pole collide; the net and the court lines are visual only. Plain Python, no USD libs.

Dimensions (NBA): rim inner diameter 18 in, rim center 15 in from the backboard face, backboard 72 x 42 in with
its bottom 6 in below the rim, free-throw line 15 ft from the backboard (4.19 m from the rim center), lane 16 ft
wide, three-point arc 23.75 ft (22 ft in the corners), rim center 63 in from the baseline.
"""

import math
import os

RIM_R = 0.2286
RIM_TUBE = 0.009
BOARD_X = 0.381  # backboard front face
FLOOR = -3.05
OUT = os.path.join(os.path.dirname(__file__), "..", "source", "basketball", "assets", "nba_hoop.usda")

WHITE, ORANGE, BLACK, GRAY, PAINT, NET = (0.95, 0.95, 0.95), (0.95, 0.4, 0.05), (0.08, 0.08, 0.08), \
    (0.25, 0.25, 0.28), (0.62, 0.22, 0.15), (0.97, 0.97, 0.97)

prims = []


def fmt(v):
    return "(" + ", ".join(f"{x:.5f}" for x in v) + ")"


def quat_z_to(d):
    """Quaternion (w, x, y, z) rotating +Z onto unit vector d."""
    z = (0.0, 0.0, 1.0)
    c = sum(a * b for a, b in zip(z, d))
    ax = (z[1] * d[2] - z[2] * d[1], z[2] * d[0] - z[0] * d[2], z[0] * d[1] - z[1] * d[0])
    w = 1.0 + c
    n = math.sqrt(w * w + sum(a * a for a in ax))
    return (w / n, ax[0] / n, ax[1] / n, ax[2] / n)


def cube(name, center, size, color, collide=False, rot_z=0.0):
    api = '( prepend apiSchemas = ["PhysicsCollisionAPI"] )' if collide else ""
    prims.append(f"""
    def Cube "{name}" {api}
    {{
        double size = 1
        color3f[] primvars:displayColor = [{fmt(color)}]
        double3 xformOp:translate = {fmt(center)}
        float xformOp:rotateZ = {rot_z:.4f}
        float3 xformOp:scale = {fmt(size)}
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateZ", "xformOp:scale"]
    }}""")


def capsule(name, center, radius, height, color, collide=False, axis="Y", rot_z=0.0, orient=None):
    api = '( prepend apiSchemas = ["PhysicsCollisionAPI"] )' if collide else ""
    if orient is not None:
        xf = f"quatf xformOp:orient = {fmt(orient)}\n        uniform token[] xformOpOrder = " \
             f'["xformOp:translate", "xformOp:orient"]'
    else:
        xf = f'float xformOp:rotateZ = {rot_z:.4f}\n        uniform token[] xformOpOrder = ' \
             f'["xformOp:translate", "xformOp:rotateZ"]'
    prims.append(f"""
    def Capsule "{name}" {api}
    {{
        uniform token axis = "{axis}"
        double height = {height:.5f}
        double radius = {radius:.5f}
        color3f[] primvars:displayColor = [{fmt(color)}]
        double3 xformOp:translate = {fmt(center)}
        {xf}
    }}""")


def floor_line(name, p0, p1, width=0.05, color=WHITE, z=FLOOR + 0.003):
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    length = math.hypot(dx, dy)
    cube(name, ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2, z), (length, width, 0.004), color,
         rot_z=math.degrees(math.atan2(dy, dx)))


def floor_arc(name, cx, radius, a0, a1, n):
    """Arc around (cx, 0) on the floor; angles in degrees measured from the -X axis toward +Y."""
    pts = [(cx - radius * math.cos(math.radians(a)), radius * math.sin(math.radians(a)))
           for a in [a0 + (a1 - a0) * i / n for i in range(n + 1)]]
    for i in range(n):
        floor_line(f"{name}_{i:02d}", pts[i], pts[i + 1])


# --- rim (32 capsules) -------------------------------------------------------------------------------------------
n_rim = 32
for i in range(n_rim):
    a = 2 * math.pi * i / n_rim
    capsule(f"Rim_{i:02d}", (RIM_R * math.cos(a), RIM_R * math.sin(a), 0.0), RIM_TUBE, 2 * math.pi * RIM_R / n_rim,
            ORANGE, collide=True, axis="Y", rot_z=math.degrees(a))
cube("RimBracket", ((RIM_R + BOARD_X) / 2, 0.0, -0.02), (BOARD_X - RIM_R, 0.10, 0.02), ORANGE, collide=True)

# --- backboard: 1.83 x 1.07 m, bottom 0.15 m below the rim -------------------------------------------------------
cube("Backboard", (BOARD_X + 0.015, 0.0, -0.15 + 1.07 / 2), (0.03, 1.83, 1.07), WHITE, collide=True)
fx = BOARD_X - 0.002  # markings on the front face
for name, c, s in [
    ("BoardEdgeTop", (fx, 0.0, 0.92 - 0.025), (0.002, 1.83, 0.05)),
    ("BoardEdgeBottom", (fx, 0.0, -0.15 + 0.025), (0.002, 1.83, 0.05)),
    ("BoardEdgeLeft", (fx, 0.89, 0.385), (0.002, 0.05, 1.07)),
    ("BoardEdgeRight", (fx, -0.89, 0.385), (0.002, 0.05, 1.07)),
    ("SquareTop", (fx, 0.0, 0.45 - 0.025), (0.002, 0.61, 0.05)),
    ("SquareBottom", (fx, 0.0, 0.0 + 0.025), (0.002, 0.61, 0.05)),
    ("SquareLeft", (fx, 0.28, 0.225), (0.002, 0.05, 0.45)),
    ("SquareRight", (fx, -0.28, 0.225), (0.002, 0.05, 0.45)),
]:
    cube(name, c, s, BLACK)

# --- support: arm from the board to a padded pole behind the baseline ---------------------------------------------
pole_x = 1.6 + 0.9
cube("SupportArm", ((BOARD_X + 0.03 + pole_x) / 2, 0.0, 0.35), (pole_x - BOARD_X - 0.03, 0.14, 0.14), GRAY,
     collide=True)
cube("Pole", (pole_x, 0.0, (0.35 + FLOOR) / 2), (0.2, 0.2, 0.35 - FLOOR + 0.07), GRAY, collide=True)
cube("PoleBase", (pole_x + 0.2, 0.0, FLOOR + 0.25), (1.0, 0.9, 0.5), (0.1, 0.15, 0.35), collide=True)

# --- net: two families of twisted strands + bottom ring (visual) --------------------------------------------------
n_net, net_depth, net_bottom_r = 12, 0.42, 0.14
bottom_ring = []
for k in range(n_net):
    for fam, twist in (("A", 25.0), ("B", -25.0)):
        a0 = math.radians(360.0 * k / n_net)
        a1 = a0 + math.radians(twist)
        p0 = (RIM_R * math.cos(a0), RIM_R * math.sin(a0), -RIM_TUBE)
        p1 = (net_bottom_r * math.cos(a1), net_bottom_r * math.sin(a1), -net_depth)
        d = [p1[i] - p0[i] for i in range(3)]
        length = math.sqrt(sum(x * x for x in d))
        u = [x / length for x in d]
        capsule(f"Net{fam}_{k:02d}", tuple((p0[i] + p1[i]) / 2 for i in range(3)), 0.004, length, NET, axis="Z",
                orient=quat_z_to(u))
for k in range(n_net):
    a = 2 * math.pi * (k + 0.5) / n_net
    capsule(f"NetRing_{k:02d}", (net_bottom_r * math.cos(a), net_bottom_r * math.sin(a), -net_depth), 0.004,
            2 * math.pi * net_bottom_r / n_net, NET, axis="Y", rot_z=math.degrees(a))

# --- half-court markings (visual) ---------------------------------------------------------------------------------
baseline_x, ft_x, lane_hw = 1.6, -4.19, 2.44
half_court = 14.33  # baseline to half-court line
cube("CourtFloor", (baseline_x - half_court / 2, 0.0, FLOOR + 0.001), (half_court, 15.24, 0.002), (0.78, 0.58, 0.36))
floor_line("HalfCourtLine", (baseline_x - half_court, -7.62), (baseline_x - half_court, 7.62))
floor_line("SidelineLeft", (baseline_x - half_court, 7.62), (baseline_x, 7.62))
floor_line("SidelineRight", (baseline_x - half_court, -7.62), (baseline_x, -7.62))
cube("Paint", ((baseline_x + ft_x) / 2, 0.0, FLOOR + 0.002), (baseline_x - ft_x, 2 * lane_hw, 0.002), PAINT)
floor_line("Baseline", (baseline_x, -7.62), (baseline_x, 7.62))
floor_line("FreeThrowLine", (ft_x, -lane_hw), (ft_x, lane_hw))
floor_line("LaneLeft", (ft_x, lane_hw), (baseline_x, lane_hw))
floor_line("LaneRight", (ft_x, -lane_hw), (baseline_x, -lane_hw))
floor_arc("FreeThrowCircle", ft_x, 1.83, -90.0, 90.0, 16)
r3, corner = 7.24, 6.71
phi = math.degrees(math.asin(corner / r3))
floor_arc("ThreePointArc", 0.0, r3, -phi, phi, 40)
x_meet = -r3 * math.cos(math.radians(phi))
floor_line("ThreeCornerLeft", (x_meet, corner), (baseline_x, corner))
floor_line("ThreeCornerRight", (x_meet, -corner), (baseline_x, -corner))
floor_arc("RestrictedArea", 0.0, 1.22, -90.0, 90.0, 12)

header = """#usda 1.0
(
    defaultPrim = "Hoop"
    metersPerUnit = 1
    upAxis = "Z"
)

# Generated by scripts/make_nba_hoop.py. Origin = rim center; place it 3.05 m above the floor.
def Xform "Hoop" (
    prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{
    bool physics:kinematicEnabled = 1
    float physics:mass = 50.0
"""
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write(header + "".join(prims) + "\n}\n")
print(f"wrote {os.path.abspath(OUT)} with {len(prims)} prims")
