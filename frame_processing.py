"""
Frame data extraction and processing

Handles extraction of multi-modal data from Aria frames:
- Pose data
- Hand tracking data
- Haptic sensor data
- Image data
"""

import logging
from typing import Dict, Any, Optional, Tuple
import numpy as np

from .schema import flatten_hand_landmarks, create_tactile_placeholders, flatten_pose_matrix

logger = logging.getLogger(__name__)


def extract_frame_data(
    session,
    timestamp_ns: int,
    step: int,
    episode_id: int,
    max_slop_ns: int,
    start_timestamp_ns: int,
    haptic_loader: Optional[Any],
    global_index: int,
    fps: float
) -> Optional[Dict[str, Any]]:
    """Extract all aligned data for a single frame

    Args:
        session: AriaSession instance
        timestamp_ns: Frame timestamp in nanoseconds
        step: Frame index within episode
        episode_id: Episode ID
        max_slop_ns: Maximum time alignment tolerance
        start_timestamp_ns: Episode start timestamp (unused, kept for compatibility)
        haptic_loader: Optional HapticDataLoader instance
        global_index: Global frame index across all episodes
        fps: Frames per second

    Returns:
        Dictionary of frame data, or None if extraction failed
    """
    # Base temporal data - use step/fps to ensure timestamps start from 0 and increment uniformly
    # This ensures perfect alignment with video frames
    time_s = step / fps
    row_data = {
        # LeRobot required index columns
        "index": np.int64(global_index),
        "episode_index": np.int64(episode_id),
        "frame_index": np.int64(step),
        "task_index": np.int64(0),  # Single task: aria_recording
        # Temporal data
        "timestamp": time_s,  # Relative time from episode start (for video sync)
        "timestamp_ns": timestamp_ns,  # Absolute timestamp (device time)
        "episode_id": np.int32(episode_id),
        "step": np.int64(step),
    }

    # Extract pose data
    pose = session.nearest_pose(timestamp_ns, max_slop_ns)
    if pose is not None:
        pose_matrix = pose.T_world_device.to_matrix()
        row_data["status.pose.world_device"] = flatten_pose_matrix(pose_matrix)
        row_data["status.pose.valid"] = True
    else:
        row_data["status.pose.world_device"] = np.full(16, np.nan, dtype=np.float32)
        row_data["status.pose.valid"] = False

    # Extract hand data
    hands_data = session.nearest_hands(timestamp_ns, max_slop_ns)
    if hands_data is not None and pose is not None:
        # Get world coordinates using the hand projector
        world_landmarks = session.hand_projector.landmarks_in_world(
            timestamp_ns, pose.T_world_device, max_slop_ns=max_slop_ns
        )

        left_landmarks = world_landmarks.get("left")
        right_landmarks = world_landmarks.get("right")

        # Store hand landmarks
        flattened, valid_left, valid_right = flatten_hand_landmarks(left_landmarks, right_landmarks)
        row_data["status.hands.landmarks_world"] = flattened
        row_data["status.hands.valid_left"] = valid_left
        row_data["status.hands.valid_right"] = valid_right
    else:
        # No hand data available
        row_data["status.hands.landmarks_world"] = np.full(63, np.nan, dtype=np.float32)
        row_data["status.hands.valid_left"] = False
        row_data["status.hands.valid_right"] = False

    # Get haptic force values if loader is available
    force_left, force_right = None, None
    if haptic_loader:
        force_left, force_right = haptic_loader.get_force_at_time(time_s)

    # Add tactile data (real or placeholders)
    tactile_data = create_tactile_placeholders(force_left, force_right)
    row_data.update(tactile_data)

    # Generate standard LeRobot columns for visualization compatibility
    # observation.state: hand landmarks (63) + device pose (16) = 79 dims
    observation_state = np.concatenate([
        row_data["status.hands.landmarks_world"],  # 63 dims
        row_data["status.pose.world_device"]       # 16 dims
    ]).astype(np.float32)
    row_data["observation.state"] = observation_state

    # action: tactile forces [left_total, right_total, left_xyz(3), right_xyz(3)] = 8 dims
    action = np.array([
        row_data["action.tactile.left.force_total"],   # 1 dim
        row_data["action.tactile.right.force_total"],  # 1 dim
        row_data["action.tactile.left.force_xyz"][0],  # 3 dims
        row_data["action.tactile.left.force_xyz"][1],
        row_data["action.tactile.left.force_xyz"][2],
        row_data["action.tactile.right.force_xyz"][0], # 3 dims
        row_data["action.tactile.right.force_xyz"][1],
        row_data["action.tactile.right.force_xyz"][2],
    ], dtype=np.float32)
    row_data["action"] = action

    return row_data


def update_processing_stats(row_data: Dict[str, Any], processed_data: Dict[str, Any]):
    """Update processing statistics based on extracted frame data

    Args:
        row_data: Extracted frame data
        processed_data: Processing statistics dictionary to update
    """
    # Update hand statistics
    if row_data.get('status.hands.valid_left') or row_data.get('status.hands.valid_right'):
        processed_data["hand_stats"]["valid"] += 1
    processed_data["hand_stats"]["total"] += 1

    # Update pose statistics
    if row_data.get('status.pose.valid'):
        processed_data["pose_stats"]["valid"] += 1
    processed_data["pose_stats"]["total"] += 1


def finalize_processing_stats(processed_data: Dict[str, Any]):
    """Finalize processing statistics after all frames are processed

    Args:
        processed_data: Processing statistics dictionary to finalize
    """
    # Calculate valid rates
    if processed_data["hand_stats"]["total"] > 0:
        processed_data["hand_stats"]["valid_rate"] = (
            processed_data["hand_stats"]["valid"] / processed_data["hand_stats"]["total"]
        )
    else:
        processed_data["hand_stats"]["valid_rate"] = 0.0

    if processed_data["pose_stats"]["total"] > 0:
        processed_data["pose_stats"]["valid_rate"] = (
            processed_data["pose_stats"]["valid"] / processed_data["pose_stats"]["total"]
        )
    else:
        processed_data["pose_stats"]["valid_rate"] = 0.0


def process_frames(
    session,
    frame_iterator,
    parquet_writer,
    progress,
    max_slop_ns: int,
    start_timestamp_ns: int = 0,
    haptic_loader: Optional[Any] = None,
    time_range_ns: Optional[Tuple[int, Optional[int]]] = None,
    episode_id: int = 0,
    global_index_start: int = 0,
    fps: float = 10.0
) -> Dict[str, Any]:
    """Process camera frames and extract aligned data

    Args:
        session: AriaSession instance
        frame_iterator: Iterator yielding (timestamp_ns, image) tuples
        parquet_writer: ParquetWriter instance
        progress: ConversionProgress instance
        max_slop_ns: Maximum time alignment tolerance
        start_timestamp_ns: Episode start timestamp
        haptic_loader: Optional HapticDataLoader instance
        time_range_ns: Optional (start_ns, end_ns) tuple to filter frames
        episode_id: Episode ID to assign to all frames
        global_index_start: Starting global index for this episode
        fps: Frames per second

    Returns:
        Dictionary with processing results and statistics
    """
    processed_data = {
        "valid_frames": 0,
        "dropped_frames": 0,
        "video_frames": [],
        "hand_stats": {"valid": 0, "total": 0},
        "pose_stats": {"valid": 0, "total": 0},
        "nan_rates": {},
    }

    step = 0
    global_index = global_index_start

    for timestamp_ns, image in frame_iterator:
        # Apply time range filter if provided
        if time_range_ns is not None:
            start_ns, end_ns = time_range_ns
            if timestamp_ns < start_ns:
                continue
            if end_ns is not None and timestamp_ns >= end_ns:
                break

        try:
            # Extract aligned data
            row_data = extract_frame_data(
                session, timestamp_ns, step, episode_id, max_slop_ns,
                start_timestamp_ns, haptic_loader, global_index, fps
            )

            if row_data is None:
                processed_data["dropped_frames"] += 1
                progress.update(valid=False)
                continue

            # Add to parquet
            parquet_writer.add_row(row_data)

            # Store video frame with relative timestamp (same as in parquet)
            relative_timestamp_s = row_data["timestamp"]
            processed_data["video_frames"].append((relative_timestamp_s, image))

            # Update statistics
            update_processing_stats(row_data, processed_data)

            processed_data["valid_frames"] += 1
            step += 1
            global_index += 1
            progress.update(valid=True)

        except Exception as e:
            logger.debug(f"Failed to process frame at {timestamp_ns}: {e}")
            processed_data["dropped_frames"] += 1
            progress.update(valid=False)
            continue

    # Finalize statistics
    finalize_processing_stats(processed_data)

    return processed_data
