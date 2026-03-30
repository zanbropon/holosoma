#!/usr/bin/env python3
"""
Simple script to extract global positions from LAFAN dataset BVH files.
"""

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import tyro
from lafan1 import extract, utils  # type: ignore[import-not-found]


def _parse_foot_end_site_offsets(
    bvh_file_path: str,
    left_foot_name: str = "LeftFoot",
    right_foot_name: str = "RightFoot",
) -> dict[str, np.ndarray]:
    """Parse End Site offsets for left/right foot joints from a BVH file.

    Returns a dict keyed by foot joint name with local offset vectors.
    """
    root_joint_re = re.compile(r"^\s*(ROOT|JOINT)\s+(\S+)")
    offset_re = re.compile(r"^\s*OFFSET\s+([\-\d\.eE]+)\s+([\-\d\.eE]+)\s+([\-\d\.eE]+)")

    joint_stack: list[str] = []
    pending_node: str | None = None
    offsets: dict[str, np.ndarray] = {}

    with open(bvh_file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = root_joint_re.match(line)
            if match:
                pending_node = match.group(2)
                continue

            if "End Site" in line:
                pending_node = "__ENDSITE__"
                continue

            if "{" in line:
                if pending_node is not None:
                    joint_stack.append(pending_node)
                    pending_node = None
                continue

            if "}" in line:
                if joint_stack:
                    joint_stack.pop()
                continue

            match = offset_re.match(line)
            if match and joint_stack and joint_stack[-1] == "__ENDSITE__" and len(joint_stack) >= 2:
                parent_joint = joint_stack[-2]
                if parent_joint in (left_foot_name, right_foot_name) and parent_joint not in offsets:
                    offsets[parent_joint] = np.array(
                        [float(match.group(1)), float(match.group(2)), float(match.group(3))], dtype=np.float64
                    )

    return offsets


def extract_global_positions(bvh_file_path, append_toe_from_end_site: bool = False):
    """
    Extract global positions from a BVH file.

    Args:
        bvh_file_path (str): Path to the BVH file

    Returns:
        dict: Dictionary containing:
            - 'positions': numpy array of shape (frames, joints, 3) with global positions
            - 'joint_names': list of joint names
            - 'parents': list of parent indices
            - 'num_frames': number of frames
            - 'num_joints': number of joints
    """
    # Read BVH file
    anim = extract.read_bvh(bvh_file_path)

    # Compute global positions using Forward Kinematics
    global_quats, global_positions = utils.quat_fk(anim.quats, anim.pos, anim.parents)

    positions = global_positions
    joint_names = list(anim.bones)

    # Optionally create pseudo toe joints from foot End Sites when explicit toe joints are absent.
    if append_toe_from_end_site and "LeftToeBase" not in joint_names and "RightToeBase" not in joint_names:
        end_site_offsets = _parse_foot_end_site_offsets(bvh_file_path)
        for foot_name, toe_name in (("LeftFoot", "LeftToeBase"), ("RightFoot", "RightToeBase")):
            if foot_name in joint_names and foot_name in end_site_offsets and toe_name not in joint_names:
                foot_idx = joint_names.index(foot_name)
                local_offset = end_site_offsets[foot_name].reshape(1, 1, 3)
                local_offset = np.repeat(local_offset, positions.shape[0], axis=0)
                rotated_offset = utils.quat_mul_vec(global_quats[:, foot_idx : foot_idx + 1, :], local_offset)
                toe_positions = positions[:, foot_idx : foot_idx + 1, :] + rotated_offset
                positions = np.concatenate([positions, toe_positions], axis=1)
                joint_names.append(toe_name)

    return {
        "positions": positions / 100,
        "joint_names": joint_names,
        "parents": anim.parents,
        "num_frames": positions.shape[0],
        "num_joints": positions.shape[1],
    }


def save_global_positions_to_npy(global_positions, output_path):
    """
    Save global positions to a .npy file.

    Args:
        global_positions (numpy.ndarray): Global positions array
        output_path (str): Output file path
    """
    np.save(output_path, global_positions)
    print(f"Saved global positions to: {output_path}")


@dataclass
class Config:
    """Configuration for extracting global positions from BVH files."""

    input_dir: str = "./lafan1/lafan"
    output_dir: str = "../demo_data/lafan"
    recursive: bool = True
    append_toe_from_end_site: bool = True


def main(cfg: Config):
    """
    Main function to extract global positions from BVH files.
    """
    input_dir = Path(cfg.input_dir)
    output_dir = Path(cfg.output_dir)

    # Check if input directory exists
    if not input_dir.exists():
        print(f"Error: Input directory {cfg.input_dir} not found!")
        print("Please run the evaluation script first to generate BVH files.")
        return

    # Get list of BVH files
    if cfg.recursive:
        bvh_files = sorted([f for f in input_dir.rglob("*.bvh") if f.is_file()])
    else:
        bvh_files = sorted([f for f in input_dir.iterdir() if f.is_file() and f.suffix == ".bvh"])

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each BVH file
    for bvh_path in bvh_files:
        rel_path = bvh_path.relative_to(input_dir)
        print(f"\nProcessing: {rel_path}")

        # Extract global positions
        result = extract_global_positions(str(bvh_path), append_toe_from_end_site=cfg.append_toe_from_end_site)

        print(f"  Frames: {result['num_frames']}")
        print(f"  Joints: {result['num_joints']}")
        print(f"  Joint names: {result['joint_names']}")

        # Save to .npy file. Keep nested path info in output filename to avoid collisions.
        output_stem = "_".join(rel_path.with_suffix("").parts)
        output_npy = output_dir / f"{output_stem}.npy"
        np.save(str(output_npy), result["positions"])


if __name__ == "__main__":
    cfg = tyro.cli(Config)
    main(cfg)
