#!/usr/bin/env python3
"""
Visualization tool for Aria LeRobot datasets with haptic and hand tracking data
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import cv2


def visualize_aria_dataset(dataset_path: str, episode_index: int = 0):
    """
    Visualize Aria dataset with video, hand tracking, and haptic data

    Args:
        dataset_path: Path to the dataset directory
        episode_index: Episode to visualize
    """

    print(f"Loading dataset from {dataset_path}...")
    dataset = LeRobotDataset(
        repo_id=Path(dataset_path).name,
        root=str(Path(dataset_path).parent)
    )

    print(f"Dataset loaded: {len(dataset)} frames")
    print(f"Available columns: {list(dataset.hf_dataset.column_names)}")

    # Get episode data
    episode_data = dataset.hf_dataset.filter(
        lambda x: x['episode_index'] == episode_index
    )

    if len(episode_data) == 0:
        print(f"No data found for episode {episode_index}")
        return

    print(f"Episode {episode_index}: {len(episode_data)} frames")

    # Setup visualization
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Video display
    ax_video = fig.add_subplot(gs[:2, :2])
    ax_video.set_title("Camera View")
    ax_video.axis('off')

    # Hand tracking plot (3D scatter)
    ax_hands = fig.add_subplot(gs[0, 2], projection='3d')
    ax_hands.set_title("Hand Landmarks")
    ax_hands.set_xlabel("X")
    ax_hands.set_ylabel("Y")
    ax_hands.set_zlabel("Z")

    # Tactile force plots
    ax_force_left = fig.add_subplot(gs[1, 2])
    ax_force_left.set_title("Left Hand Force")
    ax_force_left.set_xlabel("Frame")
    ax_force_left.set_ylabel("Force (N)")

    ax_force_right = fig.add_subplot(gs[2, 0])
    ax_force_right.set_title("Right Hand Force")
    ax_force_right.set_xlabel("Frame")
    ax_force_right.set_ylabel("Force (N)")

    # Pose info
    ax_info = fig.add_subplot(gs[2, 1])
    ax_info.axis('off')

    # Timeline
    ax_timeline = fig.add_subplot(gs[2, 2])
    ax_timeline.set_title("Timeline")
    ax_timeline.set_xlabel("Frame")
    ax_timeline.set_ylabel("Timestamp (s)")

    # Extract data for plotting
    timestamps = np.array([f['timestamp'] for f in episode_data])
    left_forces = np.array([f['action.tactile.left.force_total'] for f in episode_data])
    right_forces = np.array([f['action.tactile.right.force_total'] for f in episode_data])

    current_frame = [0]

    def update(frame_idx):
        """Update visualization for current frame"""
        if frame_idx >= len(episode_data):
            return

        frame_data = episode_data[frame_idx]
        current_frame[0] = frame_idx

        # Update video
        ax_video.clear()
        ax_video.set_title(f"Camera View - Frame {frame_idx}/{len(episode_data)}")
        ax_video.axis('off')

        # Get video frame from dataset
        try:
            sample = dataset[frame_idx]
            if 'observation.images.main' in sample:
                img = sample['observation.images.main'].numpy()
                # Convert from CHW to HWC
                if img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))
                ax_video.imshow(img)
        except Exception as e:
            ax_video.text(0.5, 0.5, f"Video error: {e}",
                         ha='center', va='center', transform=ax_video.transAxes)

        # Update hand landmarks
        ax_hands.clear()
        ax_hands.set_title("Hand Landmarks")
        ax_hands.set_xlabel("X")
        ax_hands.set_ylabel("Y")
        ax_hands.set_zlabel("Z")

        if frame_data['status.hands.valid_left']:
            landmarks = np.array(frame_data['status.hands.landmarks_world']).reshape(21, 3)
            ax_hands.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2],
                           c='blue', marker='o', s=20, label='Left Hand')

        if frame_data['status.hands.valid_right']:
            # Note: In current schema, right hand is stored separately
            # This is a placeholder - adjust based on actual data structure
            pass

        ax_hands.legend()
        ax_hands.set_xlim([-0.5, 0.5])
        ax_hands.set_ylim([-0.5, 0.5])
        ax_hands.set_zlim([-0.5, 0.5])

        # Update force plots
        ax_force_left.clear()
        ax_force_left.set_title("Left Hand Force")
        ax_force_left.plot(left_forces[:frame_idx+1], 'b-')
        ax_force_left.axvline(frame_idx, color='r', linestyle='--', alpha=0.5)
        ax_force_left.set_xlim([0, len(episode_data)])
        ax_force_left.set_ylim([0, max(np.max(left_forces), 1.0) * 1.1])
        ax_force_left.set_xlabel("Frame")
        ax_force_left.set_ylabel("Force (N)")
        ax_force_left.grid(True, alpha=0.3)

        ax_force_right.clear()
        ax_force_right.set_title("Right Hand Force")
        ax_force_right.plot(right_forces[:frame_idx+1], 'r-')
        ax_force_right.axvline(frame_idx, color='r', linestyle='--', alpha=0.5)
        ax_force_right.set_xlim([0, len(episode_data)])
        ax_force_right.set_ylim([0, max(np.max(right_forces), 1.0) * 1.1])
        ax_force_right.set_xlabel("Frame")
        ax_force_right.set_ylabel("Force (N)")
        ax_force_right.grid(True, alpha=0.3)

        # Update info text
        ax_info.clear()
        ax_info.axis('off')
        info_text = f"""
        Frame: {frame_idx} / {len(episode_data)}
        Time: {frame_data['timestamp']:.3f}s

        Hand Status:
        Left: {'Valid' if frame_data['status.hands.valid_left'] else 'Invalid'}
        Right: {'Valid' if frame_data['status.hands.valid_right'] else 'Invalid'}

        Tactile Forces:
        Left: {frame_data['action.tactile.left.force_total']:.2f} N
        Right: {frame_data['action.tactile.right.force_total']:.2f} N

        Pose: {'Valid' if frame_data['status.pose.valid'] else 'Invalid'}
        """
        ax_info.text(0.1, 0.5, info_text, fontsize=10,
                    verticalalignment='center', family='monospace')

        # Update timeline
        ax_timeline.clear()
        ax_timeline.set_title("Timeline")
        ax_timeline.plot(timestamps, 'g-')
        ax_timeline.axvline(frame_idx, color='r', linestyle='--', alpha=0.5)
        ax_timeline.axhline(frame_data['timestamp'], color='r', linestyle='--', alpha=0.5)
        ax_timeline.set_xlim([0, len(episode_data)])
        ax_timeline.set_ylim([0, max(timestamps) * 1.1])
        ax_timeline.set_xlabel("Frame")
        ax_timeline.set_ylabel("Timestamp (s)")
        ax_timeline.grid(True, alpha=0.3)

    # Create animation
    anim = FuncAnimation(fig, update, frames=len(episode_data),
                        interval=100, repeat=True)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Aria LeRobot dataset with haptics and hand tracking"
    )
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the dataset directory"
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=0,
        help="Episode index to visualize (default: 0)"
    )

    args = parser.parse_args()

    visualize_aria_dataset(args.dataset_path, args.episode)


if __name__ == "__main__":
    main()