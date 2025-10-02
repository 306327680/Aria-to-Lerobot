#!/usr/bin/env python3
"""
Load Aria dataset with NVIDIA Isaac GR00T

This script demonstrates how to load your converted Aria dataset
using the GR00T dataloader.
"""

import sys
from pathlib import Path

# Add Isaac-GR00T to path
sys.path.insert(0, '/home/echo/Isaac-GR00T')

from gr00t.data.dataset import LeRobotSingleDataset, ModalityConfig


def create_aria_modality_configs():
    """Create modality configs for Aria dataset"""
    return {
        "video": ModalityConfig(
            delta_indices=[0],  # Current frame only
            modality_keys=["video.main"]
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "state.hand_landmarks",  # 63-dim hand landmarks
                "state.gaze",            # 2-dim normalized UV gaze
                "state.device_pose"      # 14-dim device pose
            ]
        ),
        "action": ModalityConfig(
            delta_indices=[0],
            modality_keys=[
                "action.ring_tip_force_xyz",        # 3-dim ring finger force
                "action.ring_tip_force_magnitude",  # 1-dim magnitude
                "action.little_tip_force_xyz",      # 3-dim little finger force
                "action.little_tip_force_magnitude" # 1-dim magnitude
            ]
        ),
    }


def load_aria_dataset(dataset_path: str):
    """
    Load Aria dataset for GR00T training

    Args:
        dataset_path: Path to converted Aria dataset (e.g., "output_glue")

    Returns:
        LeRobotSingleDataset instance
    """
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Check for required files
    meta_dir = dataset_path / "meta"
    required_files = ["modality.json", "episodes.jsonl", "tasks.jsonl", "info.json"]
    missing = [f for f in required_files if not (meta_dir / f).exists()]

    if missing:
        raise FileNotFoundError(
            f"Missing required files in {meta_dir}: {missing}\n"
            f"Please run with --groot-format flag during conversion."
        )

    print(f"📁 Loading dataset: {dataset_path}")

    modality_configs = create_aria_modality_configs()

    dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_configs,
        embodiment_tag="new_embodiment",  # Aria as new embodiment
        video_backend="decord"
    )

    print(f"✅ Dataset loaded successfully!")
    print(f"   Samples: {len(dataset)}")
    print(f"   Dataset name: {dataset.dataset_name}")
    print(f"   Modality keys: {dict(dataset.modality_keys)}")

    return dataset


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Load Aria dataset with GR00T")
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the Aria dataset (e.g., output_glue)"
    )
    parser.add_argument(
        "--sample-index",
        type=int,
        default=None,
        help="Sample a specific index from the dataset"
    )

    args = parser.parse_args()

    # Load dataset
    dataset = load_aria_dataset(args.dataset_path)

    # Sample data if requested
    if args.sample_index is not None:
        if args.sample_index >= len(dataset):
            print(f"⚠️  Index {args.sample_index} out of range (max: {len(dataset)-1})")
            return

        print(f"\n📊 Sampling index {args.sample_index}...")
        sample = dataset[args.sample_index]

        print("\nSample structure:")
        for key, value in sample.items():
            if hasattr(value, 'shape'):
                print(f"  {key:45s} shape={str(value.shape):20s} dtype={value.dtype}")
            else:
                print(f"  {key:45s} type={type(value)}")

        # Show some data statistics
        print("\n📈 Data statistics:")
        print(f"  Video frame range: [{sample['video.main'].min()}, {sample['video.main'].max()}]")
        print(f"  Hand landmarks: {sample['state.hand_landmarks'].shape}")
        print(f"  Gaze (UV): {sample['state.gaze'][0]}")
        print(f"  Ring tip force: {sample['action.ring_tip_force_magnitude'][0]} N")
        print(f"  Little tip force: {sample['action.little_tip_force_magnitude'][0]} N")

    print("\n🎉 Dataset ready for GR00T training!")
    return dataset


if __name__ == "__main__":
    main()
