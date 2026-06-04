"""Shared workspace + coordinate-frame definitions for FetchPush-v4.

Single source of truth: both grounders (regex, LLM) and the eval harness
import from here. Edit one place to propagate everywhere.

Decisions locked here (rationale in docstring):

1. Goal dimensionality: 3D [x, y, z]. The proposal text uses 2D for human
   readability, but FetchPush-v4's `desired_goal` is shape (3,). All
   grounders emit 3D goals; the z component is fixed to TABLE_Z.

2. Workspace bounds: derived from FetchPush-v4's default goal-sampling
   region. The env samples goals centered on the gripper's initial xy
   (~1.34, 0.75) with target_range=0.15 m, yielding x in [1.19, 1.49]
   and y in [0.60, 0.90]. We mirror that range so T1 region targets
   ("upper-left corner", etc.) land inside the trained policy's
   competence zone.

3. Axis convention: robot frame. +x is forward (away from the robot
   base), +y is to the robot's left, +z is up. From this convention,
   "right" -> -y. Mental model: an observer standing behind the robot
   looking at the table. This matches MuJoCo's world frame for FetchPush.

4. Table height: TABLE_Z = 0.42 m -- the table surface where the cube
   rests. All goals are evaluated at this height.
"""

TABLE_Z = 0.42

WORKSPACE = {
    "x_min": 1.19, "x_max": 1.49,
    "y_min": 0.60, "y_max": 0.90,
    "z": TABLE_Z,
}

AXIS_CONVENTION = "robot_frame"

DIR_VECTORS = {
    "right":     (0.0, -1.0, 0.0),
    "east":      (0.0, -1.0, 0.0),
    "left":      (0.0,  1.0, 0.0),
    "west":      (0.0,  1.0, 0.0),
    "forward":   (1.0,  0.0, 0.0),
    "north":     (1.0,  0.0, 0.0),
    "back":      (-1.0, 0.0, 0.0),
    "backward":  (-1.0, 0.0, 0.0),
    "backwards": (-1.0, 0.0, 0.0),
    "south":     (-1.0, 0.0, 0.0),
}

REGIONS = {
    # "upper" = farther from the robot (larger x)
    # "lower" = closer to the robot (smaller x)
    # "left"  = robot's / viewer-behind-robot's left (larger y)
    # "right" = the other side (smaller y)
    ("upper",  "left"):  ("x_max", "y_max"),
    ("top",    "left"):  ("x_max", "y_max"),
    ("upper",  "right"): ("x_max", "y_min"),
    ("top",    "right"): ("x_max", "y_min"),
    ("lower",  "left"):  ("x_min", "y_max"),
    ("bottom", "left"):  ("x_min", "y_max"),
    ("lower",  "right"): ("x_min", "y_min"),
    ("bottom", "right"): ("x_min", "y_min"),
}


def workspace_description() -> str:
    """Human-readable workspace description for use in LLM prompts."""
    return (
        "Coordinate frame: robot frame.\n"
        "- +x is forward (away from the robot base)\n"
        "- +y is to the robot's left\n"
        "- +z is up\n"
        f"Workspace bounds (the cube and goal stay inside these):\n"
        f"- x in [{WORKSPACE['x_min']:.2f}, {WORKSPACE['x_max']:.2f}] meters\n"
        f"- y in [{WORKSPACE['y_min']:.2f}, {WORKSPACE['y_max']:.2f}] meters\n"
        f"- z = {WORKSPACE['z']:.2f} meters (table surface)\n"
        "Directional convention (top-down view, observer behind the robot):\n"
        "- 'left' means +y, 'right' means -y\n"
        "- 'forward' / 'far' / 'away' means +x; 'back' / 'toward me' means -x\n"
        "- 'upper' / 'top' / 'far edge' means +x (= x_max)\n"
        "- 'lower' / 'bottom' / 'near edge' means -x (= x_min)\n"
        "- So 'upper-left corner' = (x_max, y_max), 'lower-right corner' = (x_min, y_min)"
    )


def in_workspace(xy, slack: float = 0.0) -> bool:
    """True if [x, y] is inside the workspace (optional slack in meters)."""
    x, y = xy[0], xy[1]
    return (
        WORKSPACE["x_min"] - slack <= x <= WORKSPACE["x_max"] + slack
        and WORKSPACE["y_min"] - slack <= y <= WORKSPACE["y_max"] + slack
    )
