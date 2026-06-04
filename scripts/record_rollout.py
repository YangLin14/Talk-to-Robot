"""Record a video for one grounded FetchPush instruction.

Examples:
  python scripts/record_rollout.py --instruction-id T1_001 --grounder regex
  python scripts/record_rollout.py --instruction-id T2_001 --grounder regex --output videos/t2_001.gif
  python scripts/record_rollout.py --instruction-id T3_001 --grounder llm --variant zero-shot
  python scripts/record_rollout.py --instruction "push to the upper-left corner" --tier T1
  python scripts/record_rollout.py --instruction "push it next to the marker" --tier T3 \
    --grounder llm --reference-object marker 1.40 0.68 0.42
  python scripts/record_rollout.py --instruction "push the red cube to the right side of the blue cube" \
    --tier T3 --grounder llm --blue-cube
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from talk_to_robot.baselines.regex_grounder import ground as regex_ground
from talk_to_robot.grounding.grounder import ground as llm_ground
from talk_to_robot.workspace import WORKSPACE


DEFAULT_INSTRUCTIONS_PATH = _PROJECT_ROOT / "data" / "instructions" / "instructions.json"
DEFAULT_MODEL_PATH = _PROJECT_ROOT / "models" / "sac_her_FetchPush-v4_seed0_best.zip"
DEFAULT_ENV_ID = "FetchPush-v4"
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "videos"
DEFAULT_BLUE_CUBE_POS = [1.34, 0.75, 0.42]
DEFAULT_RED_CUBE_POS = [1.34, 0.75, 0.42]
REFERENCE_SIDE_OFFSET_M = 0.06
CUBE_COLORS = {
    "red": np.array([0.9, 0.05, 0.03, 1.0]),
    "blue": np.array([0.05, 0.15, 0.95, 1.0]),
}
TARGET_COLOR = np.array([1.0, 0.82, 0.02, 1.0])
VISUAL_CUBE_HALF_SIZE = 0.025


def load_instruction(path, instruction_id):
    data = json.loads(path.read_text())
    for entry in data["instructions"]:
        if entry["id"] == instruction_id:
            return entry
    raise SystemExit(f"Instruction id not found: {instruction_id}")


def entry_from_args(args):
    if args.instruction_id:
        return load_instruction(args.instructions, args.instruction_id)

    if not args.instruction:
        raise SystemExit("Provide either --instruction-id or --instruction.")
    if not args.tier:
        raise SystemExit("--tier is required when using --instruction.")

    return {
        "id": "custom",
        "tier": args.tier,
        "instruction": args.instruction,
        "ground_truth_goal": None,
        "requires_context": args.tier in ("T2", "T3"),
        "context": None,
    }


def sample_reference_cube_pos(seed, achieved_goal, salt):
    rng_seed = None if seed is None else seed + salt
    rng = np.random.default_rng(rng_seed)
    active_xy = np.asarray(achieved_goal[:2], dtype=float)
    x_low = WORKSPACE["x_min"] + REFERENCE_SIDE_OFFSET_M
    x_high = WORKSPACE["x_max"] - REFERENCE_SIDE_OFFSET_M
    y_low = WORKSPACE["y_min"] + REFERENCE_SIDE_OFFSET_M
    y_high = WORKSPACE["y_max"] - REFERENCE_SIDE_OFFSET_M

    for _ in range(100):
        pos = np.array([
            rng.uniform(x_low, x_high),
            rng.uniform(y_low, y_high),
        ])
        if np.linalg.norm(pos - active_xy) >= 0.08:
            return [float(pos[0]), float(pos[1]), float(WORKSPACE["z"])]

    return [float((x_low + x_high) / 2), float((y_low + y_high) / 2), float(WORKSPACE["z"])]


def context_from_args(args, entry, obs, active_color):
    context = entry.get("context")

    if args.tier == "T2" or entry.get("tier") == "T2":
        context = dict(context or {})
        if args.cube_pos:
            context["cube_pos"] = args.cube_pos
        elif "cube_pos" not in context:
            context["cube_pos"] = list(map(float, obs["achieved_goal"]))

    reference_objects = {}
    for item in args.reference_object or []:
        name, x, y, z = item
        reference_objects[name] = [float(x), float(y), float(z)]

    should_add_blue_cube = (
        args.blue_cube
        or args.blue_cube_pos is not None
        or "blue cube" in entry["instruction"].lower()
    )
    if should_add_blue_cube and active_color != "blue" and "blue cube" not in reference_objects:
        reference_objects["blue cube"] = (
            args.blue_cube_pos
            or sample_reference_cube_pos(args.seed, obs["achieved_goal"], salt=101)
        )

    should_add_red_cube = (
        args.red_cube
        or args.red_cube_pos is not None
        or "red cube" in entry["instruction"].lower()
    )
    if should_add_red_cube and active_color != "red" and "red cube" not in reference_objects:
        reference_objects["red cube"] = (
            args.red_cube_pos
            or sample_reference_cube_pos(args.seed, obs["achieved_goal"], salt=202)
        )

    if reference_objects:
        context = dict(context or {})
        context["reference_objects"] = reference_objects

    return context


def rule_ground_reference_side(instruction, context):
    """Small deterministic helper for reference-side demo instructions."""
    if not context or not context.get("reference_objects"):
        return None

    s = instruction.lower()
    for name, pos in context["reference_objects"].items():
        if name.lower() not in s:
            continue
        x, y, z = pos
        if "right side" in s or "to the right of" in s:
            goal = [x, y - REFERENCE_SIDE_OFFSET_M, z]
        elif "left side" in s or "to the left of" in s:
            goal = [x, y + REFERENCE_SIDE_OFFSET_M, z]
        elif "in front of" in s or "forward of" in s:
            goal = [x + REFERENCE_SIDE_OFFSET_M, y, z]
        elif "behind" in s or "back of" in s:
            goal = [x - REFERENCE_SIDE_OFFSET_M, y, z]
        else:
            continue

        goal[0] = float(min(max(goal[0], WORKSPACE["x_min"]), WORKSPACE["x_max"]))
        goal[1] = float(min(max(goal[1], WORKSPACE["y_min"]), WORKSPACE["y_max"]))
        goal[2] = float(WORKSPACE["z"])
        return {
            "instruction": instruction,
            "tier": "T3",
            "goal": goal,
            "grounder": "rule",
            "success": True,
            "failure_mode": None,
            "raw_match": name,
            "notes": "reference-object side rule",
        }
    return None


def prepare_t2_live_context(entry, obs):
    context = entry.get("context")
    if entry.get("tier") != "T2" or not context or "cube_pos" not in context:
        return entry, context

    live_cube_pos = list(map(float, obs["achieved_goal"]))
    old_cube = np.array(context["cube_pos"])
    old_gt = np.array(entry["ground_truth_goal"])
    offset = old_gt - old_cube

    updated_context = dict(context)
    updated_context["cube_pos"] = live_cube_pos

    updated_entry = dict(entry)
    updated_entry["ground_truth_goal"] = list(map(float, np.array(live_cube_pos) + offset))
    return updated_entry, updated_context


def write_video(frames, output_path, fps):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import imageio.v2 as imageio
    except ModuleNotFoundError as e:
        raise SystemExit("imageio is required for video writing. Install project requirements.") from e

    if output_path.suffix.lower() == ".gif":
        imageio.mimsave(output_path, frames, duration=1 / fps)
    else:
        imageio.mimsave(output_path, frames, fps=fps)


def infer_active_cube_color(instruction, requested):
    if requested != "auto":
        return requested

    s = instruction.lower()
    red_idx = s.find("red cube")
    blue_idx = s.find("blue cube")
    if red_idx == -1 and blue_idx == -1:
        return "red"
    if blue_idx == -1:
        return "red"
    if red_idx == -1:
        return "blue"
    return "red" if red_idx < blue_idx else "blue"


def try_set_active_cube_color(env, color):
    try:
        import mujoco

        model = env.unwrapped.model
        for geom_name in ("object0", "object0_geom"):
            geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
            if geom_id >= 0:
                model.geom_rgba[geom_id] = CUBE_COLORS[color]
                return
    except Exception:
        return


def try_set_target_color(env):
    try:
        import mujoco

        model = env.unwrapped.model
        candidate_names = ("target0", "target", "goal")
        for site_name in candidate_names:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_id >= 0:
                model.site_rgba[site_id] = TARGET_COLOR
                return
        for site_id in range(model.nsite):
            site_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id) or ""
            if "target" in site_name or "goal" in site_name:
                model.site_rgba[site_id] = TARGET_COLOR
                return
    except Exception:
        return


def add_reference_cube_markers(env, reference_objects):
    if not reference_objects:
        return False

    try:
        import mujoco

        renderer = getattr(env.unwrapped, "mujoco_renderer", None)
        viewer = getattr(renderer, "viewer", None)
        if viewer is None or not hasattr(viewer, "add_marker"):
            return False

        for name, pos in reference_objects.items():
            if "cube" not in name.lower():
                continue

            color = CUBE_COLORS["red"] if "red" in name.lower() else CUBE_COLORS["blue"]
            center = np.array([pos[0], pos[1], pos[2]], dtype=float)
            size = np.array([VISUAL_CUBE_HALF_SIZE] * 3, dtype=float)
            mat = np.eye(3, dtype=float).reshape(-1)

            try:
                viewer.add_marker(
                    pos=center,
                    size=size,
                    mat=mat,
                    rgba=color.astype(np.float32),
                    type=int(mujoco.mjtGeom.mjGEOM_BOX),
                    label=name,
                )
            except TypeError:
                viewer.add_marker(
                    center,
                    size,
                    mat,
                    color.astype(np.float32),
                    int(mujoco.mjtGeom.mjGEOM_BOX),
                    name,
                )
        return True
    except Exception:
        return False


def render_frame(env, obs, reference_objects, no_scene_reference_cube, no_map_overlay):
    scene_marker_added = False
    if not no_scene_reference_cube:
        scene_marker_added = add_reference_cube_markers(env, reference_objects)

    frame = env.render()
    if not no_map_overlay:
        frame = overlay_reference_map(
            frame,
            obs["achieved_goal"],
            obs["desired_goal"],
            reference_objects,
        )
    return frame, scene_marker_added


def _map_xy_to_pixels(point, origin, size):
    left, top = origin
    width, height = size
    x, y = point[0], point[1]
    px = left + (WORKSPACE["y_max"] - y) / (WORKSPACE["y_max"] - WORKSPACE["y_min"]) * width
    py = top + (WORKSPACE["x_max"] - x) / (WORKSPACE["x_max"] - WORKSPACE["x_min"]) * height
    return int(px), int(py)


def overlay_reference_map(frame, achieved_goal, desired_goal, reference_objects):
    if not reference_objects:
        return frame

    try:
        from PIL import Image, ImageDraw
    except ModuleNotFoundError:
        return frame

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")

    size = (170, 170)
    origin = (max(20, image.width - size[0] - 20), 20)
    left, top = origin
    width, height = size
    draw.rectangle(
        [left, top, left + width, top + height],
        fill=(255, 255, 255, 170),
        outline=(0, 0, 0, 220),
        width=2,
    )
    draw.text((left + 8, top + 6), "top-down map", fill=(0, 0, 0, 255))
    draw.text((left + 8, top + height - 18), "+y left", fill=(0, 0, 0, 230))
    draw.text((left + width - 58, top + 20), "+x far", fill=(0, 0, 0, 230))

    for name, pos in reference_objects.items():
        px, py = _map_xy_to_pixels(pos, origin, size)
        if "red" in name.lower():
            fill = (230, 20, 20, 255)
            text_fill = (180, 20, 20, 255)
        else:
            fill = (20, 80, 255, 255)
            text_fill = (20, 60, 200, 255)
        draw.rectangle([px - 5, py - 5, px + 5, py + 5], fill=fill)
        draw.text((px + 7, py - 7), name, fill=text_fill)

    if achieved_goal is not None:
        px, py = _map_xy_to_pixels(achieved_goal, origin, size)
        draw.ellipse([px - 5, py - 5, px + 5, py + 5], fill=(230, 20, 20, 255))

    if desired_goal is not None:
        px, py = _map_xy_to_pixels(desired_goal, origin, size)
        draw.line([px - 6, py, px + 6, py], fill=(230, 190, 0, 255), width=3)
        draw.line([px, py - 6, px, py + 6], fill=(230, 190, 0, 255), width=3)

    return np.array(image)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--instruction-id")
    parser.add_argument("--instruction", help="Custom natural-language instruction.")
    parser.add_argument("--tier", choices=["T0", "T1", "T2", "T3", "T4"],
                        help="Tier for a custom --instruction.")
    parser.add_argument("--grounder", choices=["regex", "llm"], default="regex")
    parser.add_argument("--variant", choices=["zero-shot", "few-shot", "cot"], default="zero-shot")
    parser.add_argument("--cube-pos", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="Optional cube position context for custom T2. Defaults to live achieved_goal.")
    parser.add_argument("--reference-object", nargs=4, action="append",
                        metavar=("NAME", "X", "Y", "Z"),
                        help="Reference object context for T3; may be repeated.")
    parser.add_argument("--blue-cube", action="store_true",
                        help="Add a random blue cube reference object unless --blue-cube-pos is provided.")
    parser.add_argument("--blue-cube-pos", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="Add a fixed blue cube reference object at this position.")
    parser.add_argument("--red-cube", action="store_true",
                        help="Add a random red cube reference object unless --red-cube-pos is provided.")
    parser.add_argument("--red-cube-pos", type=float, nargs=3, default=None,
                        metavar=("X", "Y", "Z"),
                        help="Add a fixed red cube reference object at this position.")
    parser.add_argument("--active-cube-color", choices=["auto", "red", "blue"], default="auto",
                        help="Color the single physical FetchPush cube. Auto picks the first color named in the instruction.")
    parser.add_argument("--no-map-overlay", action="store_true",
                        help="Do not draw the top-down reference-object overlay on the video.")
    parser.add_argument("--no-scene-reference-cube", action="store_true",
                        help="Do not add the non-colliding blue/red cube marker to the main MuJoCo scene.")
    parser.add_argument("--instructions", type=Path, default=DEFAULT_INSTRUCTIONS_PATH)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--env-id", default=DEFAULT_ENV_ID)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional random seed. Omit for a fresh random reset/reference cube each run.")
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    import gymnasium as gym
    import gymnasium_robotics
    from stable_baselines3 import SAC

    entry = entry_from_args(args)
    active_color = infer_active_cube_color(entry["instruction"], args.active_cube_color)

    gym.register_envs(gymnasium_robotics)
    env = gym.make(args.env_id, render_mode="rgb_array")
    try_set_active_cube_color(env, active_color)
    try_set_target_color(env)
    model = SAC.load(args.model_path, env=env, device=args.device)

    obs, _ = env.reset(seed=args.seed)
    if args.instruction_id:
        entry, context = prepare_t2_live_context(entry, obs)
    else:
        context = context_from_args(args, entry, obs, active_color)

    if args.grounder == "regex":
        ground_result = regex_ground(entry["instruction"], entry["tier"], context)
    else:
        ground_result = rule_ground_reference_side(entry["instruction"], context)
        if ground_result is None:
            ground_result = llm_ground(
                entry["instruction"],
                entry["tier"],
                context=context,
                prompt_variant=args.variant,
            )

    goal = ground_result.get("goal")
    if not ground_result.get("success") or goal is None:
        env.close()
        raise SystemExit(f"Grounding failed: {json.dumps(ground_result, indent=2)}")

    env.unwrapped.goal = np.array(goal, dtype=np.float32)
    obs["desired_goal"] = env.unwrapped.goal.copy()

    reference_objects = (context or {}).get("reference_objects") or {}
    # Initialize the offscreen viewer before adding scene markers.
    env.render()
    first_frame, scene_marker_added = render_frame(
        env,
        obs,
        reference_objects,
        args.no_scene_reference_cube,
        args.no_map_overlay,
    )
    frames = [first_frame]
    steps = 0
    ep_success = False
    terminated = truncated = False
    while not (terminated or truncated) and steps < args.max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        frame, marker_added = render_frame(
            env,
            obs,
            reference_objects,
            args.no_scene_reference_cube,
            args.no_map_overlay,
        )
        scene_marker_added = scene_marker_added or marker_added
        frames.append(frame)
        steps += 1
        ep_success = ep_success or bool(info.get("is_success", False))

    final_distance = float(
        np.linalg.norm(np.asarray(obs["achieved_goal"]) - np.asarray(obs["desired_goal"]))
    )

    output = args.output
    if output is None:
        suffix = "llm_" + args.variant if args.grounder == "llm" else "regex"
        output = DEFAULT_OUTPUT_DIR / f"{entry['id']}_{suffix}.mp4"

    write_video(frames, output, args.fps)
    env.close()

    print(f"instruction_id={entry['id']}")
    print(f"instruction={entry['instruction']}")
    print(f"seed={args.seed if args.seed is not None else 'random'}")
    print(f"active_physical_cube={active_color}")
    if reference_objects:
        print(f"reference_objects={reference_objects}")
        print(f"scene_reference_cube_visible={scene_marker_added}")
    print(f"goal={goal}")
    print(f"success={ep_success}")
    print(f"final_distance={final_distance:.4f}m")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
