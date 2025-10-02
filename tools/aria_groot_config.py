"""
Aria Dataset Configuration for GR00T Training

This module defines flexible configurations for selecting specific
state and action dimensions for training.
"""

from typing import List, Dict
from dataclasses import dataclass


@dataclass
class AriaStateConfig:
    """Configuration for state modalities"""

    # Hand landmark indices (each landmark has 3 coordinates: x, y, z)
    # Based on MediaPipe Hands 21-point model
    HAND_LANDMARKS = {
        "WRIST": 0,
        "THUMB_CMC": 1, "THUMB_MCP": 2, "THUMB_IP": 3, "THUMB_TIP": 4,
        "INDEX_FINGER_MCP": 5, "INDEX_FINGER_PIP": 6, "INDEX_FINGER_DIP": 7, "INDEX_FINGER_TIP": 8,
        "MIDDLE_FINGER_MCP": 9, "MIDDLE_FINGER_PIP": 10, "MIDDLE_FINGER_DIP": 11, "MIDDLE_FINGER_TIP": 12,
        "RING_FINGER_MCP": 13, "RING_FINGER_PIP": 14, "RING_FINGER_DIP": 15, "RING_FINGER_TIP": 16,
        "PINKY_MCP": 17, "PINKY_PIP": 18, "PINKY_DIP": 19, "PINKY_TIP": 20,
    }

    @staticmethod
    def get_landmark_indices(landmark_names: List[str]) -> List[int]:
        """Get flattened indices for specified landmarks

        Args:
            landmark_names: List of landmark names (e.g., ["THUMB_TIP", "INDEX_FINGER_TIP"])

        Returns:
            List of indices in the 63-dim hand_landmarks array
        """
        indices = []
        for name in landmark_names:
            if name not in AriaStateConfig.HAND_LANDMARKS:
                raise ValueError(f"Unknown landmark: {name}")
            landmark_idx = AriaStateConfig.HAND_LANDMARKS[name]
            # Each landmark has 3 coordinates (x, y, z)
            indices.extend([landmark_idx * 3, landmark_idx * 3 + 1, landmark_idx * 3 + 2])
        return indices


@dataclass
class AriaActionConfig:
    """Configuration for action modalities"""

    # Action array structure (8 dims total):
    # [0:3]   ring_tip_force_xyz
    # [3:4]   ring_tip_force_magnitude
    # [4:7]   little_tip_force_xyz
    # [7:8]   little_tip_force_magnitude

    ACTION_DIMS = {
        "ring_tip_force_xyz": [0, 1, 2],
        "ring_tip_force_magnitude": [3],
        "little_tip_force_xyz": [4, 5, 6],
        "little_tip_force_magnitude": [7],
    }

    @staticmethod
    def get_action_indices(action_names: List[str]) -> List[int]:
        """Get indices for specified action modalities

        Args:
            action_names: List of action names (e.g., ["ring_tip_force_magnitude", "little_tip_force_magnitude"])

        Returns:
            List of indices in the 8-dim action array
        """
        indices = []
        for name in action_names:
            if name not in AriaActionConfig.ACTION_DIMS:
                raise ValueError(f"Unknown action: {name}")
            indices.extend(AriaActionConfig.ACTION_DIMS[name])
        return indices


# ============================================================================
# Preset Configurations
# ============================================================================

# Example 1: Thumb + Index Finger Tips (pinch grasp)
PINCH_GRASP_CONFIG = {
    "state": {
        "landmarks": ["THUMB_TIP", "INDEX_FINGER_TIP"],  # 6 dims (2 landmarks × 3)
        "use_gaze": False,
        "use_pose": False,
    },
    "action": {
        "modalities": ["ring_tip_force_magnitude", "little_tip_force_magnitude"],  # 2 dims
    }
}

# Example 2: Full Hand State
FULL_HAND_CONFIG = {
    "state": {
        "landmarks": None,  # Use all 21 landmarks (63 dims)
        "use_gaze": True,   # Add gaze (2 dims)
        "use_pose": True,   # Add device pose (14 dims)
    },
    "action": {
        "modalities": [
            "ring_tip_force_xyz",
            "ring_tip_force_magnitude",
            "little_tip_force_xyz",
            "little_tip_force_magnitude"
        ],  # All 8 dims
    }
}

# Example 3: Wrist + Fingertips
FINGERTIPS_CONFIG = {
    "state": {
        "landmarks": [
            "WRIST",
            "THUMB_TIP",
            "INDEX_FINGER_TIP",
            "MIDDLE_FINGER_TIP",
            "RING_FINGER_TIP",
            "PINKY_TIP"
        ],  # 18 dims (6 landmarks × 3)
        "use_gaze": True,
        "use_pose": False,
    },
    "action": {
        "modalities": ["ring_tip_force_magnitude", "little_tip_force_magnitude"],
    }
}


def get_state_indices(config: Dict) -> List[int]:
    """Get combined state indices based on config

    Args:
        config: Configuration dict with 'landmarks', 'use_gaze', 'use_pose'

    Returns:
        List of indices to extract from observation.state (79 dims)
    """
    indices = []

    # Hand landmarks [0:63]
    if config.get("landmarks") is None:
        # Use all hand landmarks
        indices.extend(range(0, 63))
    elif config.get("landmarks"):
        # Use specific landmarks
        landmark_indices = AriaStateConfig.get_landmark_indices(config["landmarks"])
        indices.extend(landmark_indices)

    # Gaze [63:65]
    if config.get("use_gaze", False):
        indices.extend([63, 64])

    # Device pose [65:79]
    if config.get("use_pose", False):
        indices.extend(range(65, 79))

    return indices


def get_action_indices(config: Dict) -> List[int]:
    """Get action indices based on config

    Args:
        config: Configuration dict with 'modalities'

    Returns:
        List of indices to extract from action (8 dims)
    """
    if not config.get("modalities"):
        # Use all action dims
        return list(range(8))

    return AriaActionConfig.get_action_indices(config["modalities"])


def print_config_summary(config_name: str, config: Dict):
    """Print a summary of the configuration"""
    print(f"\n{'='*60}")
    print(f"Configuration: {config_name}")
    print(f"{'='*60}")

    state_indices = get_state_indices(config["state"])
    action_indices = get_action_indices(config["action"])

    print(f"\nState dimensions: {len(state_indices)}")
    print(f"  Indices: {state_indices[:10]}{'...' if len(state_indices) > 10 else ''}")

    if config["state"].get("landmarks"):
        print(f"  Landmarks: {', '.join(config['state']['landmarks'])}")
    else:
        print(f"  Landmarks: All 21 landmarks")

    print(f"  Gaze: {'Yes' if config['state'].get('use_gaze') else 'No'}")
    print(f"  Pose: {'Yes' if config['state'].get('use_pose') else 'No'}")

    print(f"\nAction dimensions: {len(action_indices)}")
    print(f"  Indices: {action_indices}")
    print(f"  Modalities: {', '.join(config['action'].get('modalities', ['All']))}")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    # Demo all preset configurations
    print("Aria Dataset Configuration Presets")

    print_config_summary("PINCH_GRASP_CONFIG", PINCH_GRASP_CONFIG)
    print_config_summary("FULL_HAND_CONFIG", FULL_HAND_CONFIG)
    print_config_summary("FINGERTIPS_CONFIG", FINGERTIPS_CONFIG)
