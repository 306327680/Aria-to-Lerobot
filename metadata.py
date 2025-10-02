"""
Metadata generation for LeRobot datasets

Handles creation of all metadata files required by LeRobot v3.0:
- info.json
- tasks.parquet
- episodes/ (parquet files)
- stats.json
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
import numpy as np
import pandas as pd

from .schema import LeRobotSchema

logger = logging.getLogger(__name__)


def create_tasks_parquet(meta_dir: Path) -> Path:
    """Create tasks.parquet metadata file (v3.0 format)

    Args:
        meta_dir: Path to meta directory

    Returns:
        Path to created tasks.parquet file
    """
    tasks_file = meta_dir / "tasks.parquet"
    task_data = pd.DataFrame([{
        "task_index": 0,
        "task": "aria_recording",
        "task_description": "Project Aria multi-modal recording converted to LeRobot format"
    }])
    task_data.to_parquet(tasks_file, index=False)

    logger.info("Created tasks.parquet (v3.0 format)")
    return tasks_file


def create_episodes_parquet(meta_dir: Path, chunk_results: List[Dict], num_episodes: int,
                           video_keys: List[str] = None, fps: float = 10.0) -> Path:
    """Create episodes parquet files (v3.0 format)

    Args:
        meta_dir: Path to meta directory
        chunk_results: List of chunk processing results
        num_episodes: Total number of episodes
        video_keys: List of video keys (e.g., ['observation.images.main'])

    Returns:
        Path to created episodes directory
    """
    episodes_dir = meta_dir / "episodes"
    episodes_dir.mkdir(exist_ok=True, parents=True)

    if video_keys is None:
        video_keys = ["observation.images.main"]

    # Build episode records with v3.0 required fields
    episode_records = []
    cumulative_frames = 0

    for i, chunk_result in enumerate(chunk_results):
        # Get frame count from chunk result
        frame_count = chunk_result.get('result', {}).get('processing', {}).get('valid_frames', 0)

        # Calculate episode duration in seconds
        episode_duration_s = float(frame_count) / fps if frame_count > 0 else 0.0

        episode_data = {
            "episode_index": i,
            "task_index": 0,
            "task": "aria_recording",
            "length": frame_count,
            # v3.0 required fields for data files
            "data/chunk_index": 0,  # All in chunk 0 for now
            "data/file_index": i,   # One file per episode
            "dataset_from_index": cumulative_frames,
            "dataset_to_index": cumulative_frames + frame_count,
        }

        # Add video file indices and timestamps for each video key
        # Timestamps represent position in the concatenated video file (in seconds)
        # For single-episode files: from=0.0, to=episode_duration
        for vid_key in video_keys:
            episode_data[f"videos/{vid_key}/chunk_index"] = 0
            episode_data[f"videos/{vid_key}/file_index"] = i
            episode_data[f"videos/{vid_key}/from_timestamp"] = 0.0  # Start of this video file
            episode_data[f"videos/{vid_key}/to_timestamp"] = episode_duration_s  # End of this video file

        episode_records.append(episode_data)
        cumulative_frames += frame_count

    # Write to parquet with v3.0 nested structure: meta/episodes/chunk-000/file-000.parquet
    episodes_df = pd.DataFrame(episode_records)
    chunk_dir = episodes_dir / "chunk-000"
    chunk_dir.mkdir(exist_ok=True, parents=True)
    episodes_file = chunk_dir / "file-000.parquet"
    episodes_df.to_parquet(episodes_file, index=False)

    logger.info(f"Created episodes/chunk-000/file-000.parquet with {num_episodes} episodes (v3.0 format)")
    return episodes_dir


def create_stats_json(meta_dir: Path) -> Path:
    """Create stats.json metadata file (v3.0 format)

    Creates empty global stats that will be computed by LeRobot on first load.

    Args:
        meta_dir: Path to meta directory

    Returns:
        Path to created stats.json file
    """
    stats_file = meta_dir / "stats.json"
    # Empty stats - will be computed by LeRobot on first load
    stats_data = {}

    with open(stats_file, 'w') as f:
        json.dump(stats_data, f, indent=2)

    logger.info("Created stats.json (v3.0 format, stats will be computed by LeRobot)")
    return stats_file


def update_consolidated_metadata(
    out_dir: Path,
    chunk_results: List[Dict],
    session_info: Dict[str, Any],
    fps: float,
    num_episodes: int,
    total_frames: int,
    total_parquet: int,
    total_video: int,
    schema: LeRobotSchema
) -> Dict[str, Path]:
    """Update all metadata files after chunk processing completes

    Args:
        out_dir: Output directory
        chunk_results: List of chunk processing results
        session_info: Aria session information
        fps: Frames per second
        num_episodes: Total number of episodes
        total_frames: Total number of frames across all episodes
        total_parquet: Total number of parquet files
        total_video: Total number of video files
        schema: LeRobot schema to use

    Returns:
        Dictionary mapping metadata file names to paths
    """
    meta_dir = out_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Updating consolidated metadata...")

    # Use Aria RGB camera default resolution
    video_shape = (1408, 1408, 3)

    # Update info.json
    info_data = schema.get_info_json(
        fps=fps,
        episode_count=num_episodes,
        total_frames=total_frames,
        video_shape=video_shape,
        aria_session_info=session_info,
        conversion_time=time.time(),
        video_shards=total_video,
        parquet_shards=total_parquet,
    )

    info_file = meta_dir / "info.json"
    with open(info_file, 'w') as f:
        json.dump(info_data, f, indent=2)
    logger.info(f"Updated info.json with {num_episodes} episodes, {total_frames} total frames")

    # Create metadata files (v3.0 format)
    tasks_file = create_tasks_parquet(meta_dir)
    episodes_dir = create_episodes_parquet(meta_dir, chunk_results, num_episodes,
                                           video_keys=schema.video_columns, fps=fps)
    stats_file = create_stats_json(meta_dir)

    logger.info("Consolidated metadata update complete (v3.0 format)")

    return {
        "info.json": info_file,
        "tasks.parquet": tasks_file,
        "episodes": episodes_dir,
        "stats.json": stats_file,
    }


def write_single_episode_metadata(
    structure: Dict[str, Path],
    schema: LeRobotSchema,
    session_info: Dict[str, Any],
    fps: float,
    parquet_metadata: Dict[str, Any],
    video_metadata: List[Dict[str, Any]],
    episode_id: int,
    total_frames: int
) -> List[str]:
    """Write metadata files for single episode (non-chunked) mode

    Args:
        structure: Output directory structure
        schema: LeRobot schema
        session_info: Aria session information
        fps: Frames per second
        parquet_metadata: Parquet file metadata
        video_metadata: Video file metadata
        episode_id: Episode ID
        total_frames: Total number of frames

    Returns:
        List of written metadata file paths
    """
    written_files = []

    # Get video shape from first video metadata
    if video_metadata:
        video_shape = video_metadata[0].get("video_shape", (1408, 1408, 3))
    else:
        video_shape = (1408, 1408, 3)

    # Create info.json
    info_data = schema.get_info_json(
        fps=fps,
        episode_count=1,  # Single episode
        total_frames=total_frames,
        video_shape=video_shape,
        aria_session_info=session_info,
        conversion_time=time.time(),
        video_shards=len(video_metadata),
        parquet_shards=len(parquet_metadata.get("shards", [])),
    )

    info_file = structure["meta"] / "info.json"
    with open(info_file, 'w') as f:
        json.dump(info_data, f, indent=2)
    written_files.append(str(info_file))

    # Create tasks.parquet (v3.0)
    tasks_file = create_tasks_parquet(structure["meta"])
    written_files.append(str(tasks_file))

    # Create episodes parquet (v3.0)
    chunk_results = [{
        'result': {
            'processing': {
                'valid_frames': total_frames
            }
        }
    }]
    video_keys = schema.video_columns  # Already a list
    episodes_dir = create_episodes_parquet(structure["meta"], chunk_results, 1, video_keys, fps=fps)
    written_files.append(str(episodes_dir))

    # Create stats.json (v3.0)
    stats_file = create_stats_json(structure["meta"])
    written_files.append(str(stats_file))

    logger.info(f"Written metadata files: {written_files}")
    return written_files


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj
