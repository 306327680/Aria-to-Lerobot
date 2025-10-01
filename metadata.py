"""
Metadata generation for LeRobot datasets

Handles creation of all metadata files required by LeRobot v3.0:
- info.json
- tasks.jsonl
- episodes.jsonl
- episodes_stats.jsonl
- stats.json
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import time
import numpy as np

from .schema import LeRobotSchema

logger = logging.getLogger(__name__)


def create_tasks_jsonl(meta_dir: Path) -> Path:
    """Create tasks.jsonl metadata file

    Args:
        meta_dir: Path to meta directory

    Returns:
        Path to created tasks.jsonl file
    """
    tasks_file = meta_dir / "tasks.jsonl"
    with open(tasks_file, 'w') as f:
        task_info = {
            "task_index": 0,
            "task": "aria_recording",
            "task_description": "Project Aria multi-modal recording converted to LeRobot format"
        }
        f.write(json.dumps(task_info) + "\n")

    logger.info("Created tasks.jsonl")
    return tasks_file


def create_episodes_jsonl(meta_dir: Path, chunk_results: List[Dict], num_episodes: int) -> Path:
    """Create episodes.jsonl metadata file

    Args:
        meta_dir: Path to meta directory
        chunk_results: List of chunk processing results
        num_episodes: Total number of episodes

    Returns:
        Path to created episodes.jsonl file
    """
    episodes_file = meta_dir / "episodes.jsonl"
    with open(episodes_file, 'w') as f:
        for i, chunk_result in enumerate(chunk_results):
            # Get frame count from chunk result
            frame_count = chunk_result.get('result', {}).get('processing', {}).get('valid_frames', 0)
            episode_data = {
                "episode_index": i,
                "task_index": 0,
                "task": "aria_recording",
                "length": frame_count,
            }
            f.write(json.dumps(episode_data) + "\n")

    logger.info(f"Created episodes.jsonl with {num_episodes} episodes")
    return episodes_file


def create_episodes_stats_jsonl(meta_dir: Path, num_episodes: int) -> Path:
    """Create episodes_stats.jsonl metadata file

    Creates empty stats entries that will be computed by LeRobot on first load.

    Args:
        meta_dir: Path to meta directory
        num_episodes: Total number of episodes

    Returns:
        Path to created episodes_stats.jsonl file
    """
    episodes_stats_file = meta_dir / "episodes_stats.jsonl"
    with open(episodes_stats_file, 'w') as f:
        for i in range(num_episodes):
            # Empty stats - will be computed by LeRobot
            stats_data = {
                "episode_index": i,
                "stats": {}  # Empty dict - LeRobot will fill this in
            }
            f.write(json.dumps(stats_data) + "\n")

    logger.info(f"Created episodes_stats.jsonl with {num_episodes} episodes (stats will be computed by LeRobot)")
    return episodes_stats_file


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

    # Create metadata files
    tasks_file = create_tasks_jsonl(meta_dir)
    episodes_file = create_episodes_jsonl(meta_dir, chunk_results, num_episodes)
    episodes_stats_file = create_episodes_stats_jsonl(meta_dir, num_episodes)

    logger.info("Consolidated metadata update complete")

    return {
        "info.json": info_file,
        "tasks.jsonl": tasks_file,
        "episodes.jsonl": episodes_file,
        "episodes_stats.jsonl": episodes_stats_file,
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

    # Create tasks.jsonl
    tasks_file = create_tasks_jsonl(structure["meta"])
    written_files.append(str(tasks_file))

    # Create episodes.jsonl
    episodes_file = structure["meta"] / "episodes.jsonl"
    with open(episodes_file, 'w') as f:
        episode_data = {
            "episode_index": episode_id,
            "task_index": 0,
            "task": "aria_recording",
            "length": total_frames,
        }
        f.write(json.dumps(episode_data) + "\n")
    written_files.append(str(episodes_file))

    # Create episodes_stats.jsonl
    episodes_stats_file = structure["meta"] / "episodes_stats.jsonl"
    with open(episodes_stats_file, 'w') as f:
        stats_data = {
            "episode_index": episode_id,
            "stats": {}
        }
        f.write(json.dumps(stats_data) + "\n")
    written_files.append(str(episodes_stats_file))

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
