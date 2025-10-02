#!/usr/bin/env python3
"""
Example training script using PINCH_GRASP_CONFIG

This demonstrates how to:
1. Load the Aria dataset with GR00T
2. Select specific dimensions (thumb + index finger tips, two force magnitudes)
3. Prepare data for training
"""

import sys
sys.path.insert(0, '/home/echo/Isaac-GR00T')

# Add aria2lerobot/tools to path
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from pathlib import Path
import torch
import numpy as np
from gr00t.data.dataset import LeRobotSingleDataset, ModalityConfig
from aria_groot_config import PINCH_GRASP_CONFIG, get_state_indices, get_action_indices

# ============================================================================
# 1. Configuration
# ============================================================================

print("=" * 60)
print("Aria GR00T Training Example - Pinch Grasp Configuration")
print("=" * 60)

# Use PINCH_GRASP_CONFIG for thumb + index finger
config = PINCH_GRASP_CONFIG

state_indices = get_state_indices(config["state"])
action_indices = get_action_indices(config["action"])

print(f"\nConfiguration:")
print(f"  State dims: {len(state_indices)} (landmarks: {config['state']['landmarks']})")
print(f"  State indices: {state_indices}")
print(f"  Action dims: {len(action_indices)} (modalities: {config['action']['modalities']})")
print(f"  Action indices: {action_indices}")

# ============================================================================
# 2. Dataset Loading
# ============================================================================

dataset_path = Path("output_glue")

print(f"\n📁 Loading dataset: {dataset_path}")

modality_configs = {
    "video": ModalityConfig(
        delta_indices=[0],  # Current frame only
        modality_keys=["video.main"]
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["state.hand_landmarks"]  # Load full 63-dim, will select later
    ),
    "action": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "action.ring_tip_force_magnitude",
            "action.little_tip_force_magnitude"
        ]
    ),
}

dataset = LeRobotSingleDataset(
    dataset_path=dataset_path,
    modality_configs=modality_configs,
    embodiment_tag="new_embodiment",
    video_backend="decord"
)

print(f"✅ Dataset loaded: {len(dataset)} samples")

# ============================================================================
# 3. Sample Data and Dimension Selection
# ============================================================================

print(f"\n📊 Sampling first 5 frames...")

for idx in range(min(5, len(dataset))):
    sample = dataset[idx]

    # Extract data
    video = sample["video.main"]  # (1, H, W, 3)
    state_full = sample["state.hand_landmarks"]  # (1, 63)
    action_ring = sample["action.ring_tip_force_magnitude"]  # (1, 1)
    action_little = sample["action.little_tip_force_magnitude"]  # (1, 1)

    # Select specific dimensions (thumb + index finger)
    # state_indices = [12, 13, 14, 24, 25, 26] for THUMB_TIP and INDEX_FINGER_TIP
    state_selected = state_full[:, state_indices]  # (1, 6)

    # Combine action forces (convert to numpy for concatenation)
    action = np.concatenate([action_ring, action_little], axis=-1)  # (1, 2)

    print(f"\nSample {idx}:")
    print(f"  Video shape: {video.shape}")
    print(f"  State (thumb+index): {state_selected.squeeze().tolist()}")
    print(f"  Action (ring, little): {action.squeeze().tolist()}")

# ============================================================================
# 4. Training Loop Template
# ============================================================================

print(f"\n" + "=" * 60)
print("Training Loop Template")
print("=" * 60)

print("""
# Pseudo-code for training loop:

for epoch in range(num_epochs):
    for batch_idx in range(len(dataset)):
        sample = dataset[batch_idx]

        # Extract and select dimensions
        video = sample["video.main"]
        state = sample["state.hand_landmarks"][:, state_indices]  # 6 dims
        action = np.concatenate([
            sample["action.ring_tip_force_magnitude"],
            sample["action.little_tip_force_magnitude"]
        ], axis=-1)  # 2 dims

        # Forward pass
        # pred_action = model(video, state)

        # Compute loss
        # loss = loss_fn(pred_action, action)

        # Backward pass
        # optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()
""")

print("=" * 60)
print("🎉 Example complete! Ready for GR00T training.")
print("=" * 60)
