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


def create_tasks_parquet(meta_dir: Path, task_name: str = "aria_recording",
                        task_description: str = "Project Aria multi-modal recording converted to LeRobot format") -> Path:
    """Create tasks.parquet metadata file (v3.0 format)

    Args:
        meta_dir: Path to meta directory
        task_name: Name of the task (default: "aria_recording")
        task_description: Description of the task

    Returns:
        Path to created tasks.parquet file
    """
    tasks_file = meta_dir / "tasks.parquet"
    task_data = pd.DataFrame([{
        "task_index": 0,
        "task": task_name,
        "task_description": task_description
    }])
    task_data.to_parquet(tasks_file, index=False)

    logger.info(f"Created tasks.parquet (v3.0 format) with task: {task_name}")
    return tasks_file


def create_episodes_parquet(meta_dir: Path, chunk_results: List[Dict], num_episodes: int,
                           video_keys: List[str] = None, fps: float = 10.0,
                           task_name: str = "aria_recording") -> Path:
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
            "task": task_name,
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


def create_stats_json(meta_dir: Path, schema=None, data_dir: Path = None) -> Path:
    """Create stats.json metadata file (v3.0 format)

    Computes statistics for all state and action columns required by GR00T.

    Args:
        meta_dir: Path to meta directory
        schema: LeRobot schema (optional, for getting column names)
        data_dir: Path to data directory (optional, for computing stats)

    Returns:
        Path to created stats.json file
    """
    stats_file = meta_dir / "stats.json"

    # If schema and data_dir provided, compute actual statistics
    if schema is not None and data_dir is not None and data_dir.exists():
        try:
            # Find all parquet files
            parquet_files = sorted(data_dir.glob("chunk-*/episode_*.parquet"))

            if parquet_files:
                logger.info(f"Computing statistics from {len(parquet_files)} parquet files...")

                # Read first file to get column structure
                sample_df = pd.read_parquet(parquet_files[0])

                stats_data = {}

                # Compute stats for observation.state
                if 'observation.state' in sample_df.columns:
                    state_data = []
                    for pf in parquet_files:
                        df = pd.read_parquet(pf)
                        state_values = np.vstack(df['observation.state'].values)
                        state_data.append(state_values)

                    all_state = np.vstack(state_data)

                    # Compute mean and std, handling NaN values
                    stats_data['observation.state'] = {
                        'mean': convert_numpy_types(np.nanmean(all_state, axis=0)),
                        'std': convert_numpy_types(np.nanstd(all_state, axis=0)),
                        'min': convert_numpy_types(np.nanmin(all_state, axis=0)),
                        'max': convert_numpy_types(np.nanmax(all_state, axis=0)),
                        'q01': convert_numpy_types(np.nanpercentile(all_state, 1, axis=0)),
                        'q99': convert_numpy_types(np.nanpercentile(all_state, 99, axis=0)),
                    }

                # Compute stats for action
                if 'action' in sample_df.columns:
                    action_data = []
                    for pf in parquet_files:
                        df = pd.read_parquet(pf)
                        action_values = np.vstack(df['action'].values)
                        action_data.append(action_values)

                    all_action = np.vstack(action_data)

                    stats_data['action'] = {
                        'mean': convert_numpy_types(np.mean(all_action, axis=0)),
                        'std': convert_numpy_types(np.std(all_action, axis=0)),
                        'min': convert_numpy_types(np.min(all_action, axis=0)),
                        'max': convert_numpy_types(np.max(all_action, axis=0)),
                        'q01': convert_numpy_types(np.percentile(all_action, 1, axis=0)),
                        'q99': convert_numpy_types(np.percentile(all_action, 99, axis=0)),
                    }

                logger.info(f"Computed statistics for {len(stats_data)} modalities")
            else:
                logger.warning("No parquet files found, creating empty stats")
                stats_data = {}
        except Exception as e:
            logger.warning(f"Failed to compute statistics: {e}, creating empty stats")
            stats_data = {}
    else:
        # Empty stats - will be computed by LeRobot on first load
        stats_data = {}

    with open(stats_file, 'w') as f:
        json.dump(stats_data, f, indent=2)

    if stats_data:
        logger.info("Created stats.json with computed statistics")
    else:
        logger.info("Created empty stats.json (stats will be computed by LeRobot)")

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
    schema: LeRobotSchema,
    task_name: str = "aria_recording",
    task_description: str = "Project Aria multi-modal recording converted to LeRobot format",
    create_groot_modality: bool = False
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
    tasks_file = create_tasks_parquet(meta_dir, task_name, task_description)
    episodes_dir = create_episodes_parquet(meta_dir, chunk_results, num_episodes,
                                           video_keys=schema.video_columns, fps=fps, task_name=task_name)
    stats_file = create_stats_json(meta_dir, schema=schema, data_dir=out_dir / "data")

    # Create GR00T modality.json if requested
    result = {
        "info.json": info_file,
        "tasks.parquet": tasks_file,
        "episodes": episodes_dir,
        "stats.json": stats_file,
    }

    if create_groot_modality:
        modality_file = create_modality_json(meta_dir, schema)
        result["modality.json"] = modality_file

        # Also create JSONL files for GR00T
        tasks_jsonl, episodes_jsonl = create_groot_jsonl_metadata(meta_dir)
        result["tasks.jsonl"] = tasks_jsonl
        result["episodes.jsonl"] = episodes_jsonl

        logger.info("Consolidated metadata update complete (v3.0 format + full GR00T compatibility)")
    else:
        logger.info("Consolidated metadata update complete (v3.0 format)")

    return result


def write_single_episode_metadata(
    structure: Dict[str, Path],
    schema: LeRobotSchema,
    session_info: Dict[str, Any],
    fps: float,
    parquet_metadata: Dict[str, Any],
    video_metadata: List[Dict[str, Any]],
    episode_id: int,
    total_frames: int,
    task_name: str = "aria_recording",
    task_description: str = "Project Aria multi-modal recording converted to LeRobot format",
    create_groot_modality: bool = False
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
    tasks_file = create_tasks_parquet(structure["meta"], task_name, task_description)
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
    episodes_dir = create_episodes_parquet(structure["meta"], chunk_results, 1, video_keys, fps=fps, task_name=task_name)
    written_files.append(str(episodes_dir))

    # Create stats.json (v3.0)
    stats_file = create_stats_json(structure["meta"], schema=schema, data_dir=structure["data"])
    written_files.append(str(stats_file))

    # Create GR00T compatibility files if requested
    if create_groot_modality:
        modality_file = create_modality_json(structure["meta"], schema)
        written_files.append(str(modality_file))

        # Also create JSONL files for GR00T
        tasks_jsonl, episodes_jsonl = create_groot_jsonl_metadata(structure["meta"])
        written_files.append(str(tasks_jsonl))
        written_files.append(str(episodes_jsonl))

        logger.info(f"Written metadata files (including full GR00T compatibility): {written_files}")
    else:
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


def create_modality_json(meta_dir: Path, schema: LeRobotSchema) -> Path:
    """Create modality.json metadata file for NVIDIA GR00T compatibility

    This file provides detailed metadata about state and action modalities,
    enabling GR00T to interpret concatenated float32 arrays as distinct fields.

    Args:
        meta_dir: Path to meta directory
        schema: LeRobot schema containing column definitions

    Returns:
        Path to created modality.json file
    """
    modality_file = meta_dir / "modality.json"

    # Analyze observation.state structure (79 elements)
    # Based on Aria data: [hand_landmarks (63), gaze (2), device_pose (14)]
    # Total: 63 + 2 + 14 = 79 elements
    # Note: hand_landmarks may be NaN when hands are not detected

    state_modalities = {
        "hand_landmarks": {
            "original_key": "observation.state",
            "start": 0,
            "end": 63,
            "description": "Hand landmarks in world coordinates (21 joints * 3 coords). May contain NaN when hands not detected."
        },
        "gaze": {
            "original_key": "observation.state",
            "start": 63,
            "end": 65,
            "description": "Eye gaze projection in normalized UV coordinates [0-1] on RGB camera image plane"
        },
        "device_pose": {
            "original_key": "observation.state",
            "start": 65,
            "end": 79,
            "description": "Device pose: 4x4 transformation matrix (world_T_device) flattened, last 14 elements [rotation(9) + translation(3) + bottom_row(2)]"
        }
    }

    # Analyze action structure (8 elements)
    # Based on tactile sensor CSV: Ring Tip + Little Tip forces
    # Format: [ring_fx, ring_fy, ring_fz, ring_mag, little_fx, little_fy, little_fz, little_mag]
    action_modalities = {
        "ring_tip_force_xyz": {
            "original_key": "action",
            "start": 0,
            "end": 3,
            "description": "Ring finger tip tactile force vector (fx, fy, fz) from sensor"
        },
        "ring_tip_force_magnitude": {
            "original_key": "action",
            "start": 3,
            "end": 4,
            "description": "Ring finger tip tactile force magnitude"
        },
        "little_tip_force_xyz": {
            "original_key": "action",
            "start": 4,
            "end": 7,
            "description": "Little finger tip tactile force vector (fx, fy, fz) from sensor"
        },
        "little_tip_force_magnitude": {
            "original_key": "action",
            "start": 7,
            "end": 8,
            "description": "Little finger tip tactile force magnitude"
        }
    }

    # Video key mapping (standardize camera names)
    video_modalities = {}
    for vid_col in schema.video_columns:
        # Extract camera name from column (e.g., "observation.images.main" -> "main")
        camera_name = vid_col.split('.')[-1]
        video_modalities[camera_name] = {
            "original_key": vid_col
        }

    # Annotation mapping (map task_index to human-readable task description)
    annotation_modalities = {
        "human.task_description": {
            "original_key": "task_index",
            "description": "Task description annotation from tasks.parquet"
        }
    }

    modality_data = {
        "state": state_modalities,
        "action": action_modalities,
        "video": video_modalities,
        "annotation": annotation_modalities
    }

    with open(modality_file, 'w') as f:
        json.dump(modality_data, f, indent=2)

    logger.info(f"Created modality.json for GR00T compatibility")
    return modality_file


def create_groot_jsonl_metadata(meta_dir: Path) -> tuple[Path, Path]:
    """Create JSONL metadata files required by NVIDIA Isaac GR00T

    GR00T requires tasks.jsonl and episodes.jsonl instead of Parquet format.
    This function converts the Parquet files to JSONL for GR00T compatibility.

    Args:
        meta_dir: Path to meta directory

    Returns:
        Tuple of (tasks_jsonl_path, episodes_jsonl_path)
    """
    import json

    # Convert tasks.parquet to tasks.jsonl
    tasks_parquet = meta_dir / "tasks.parquet"
    tasks_jsonl = meta_dir / "tasks.jsonl"

    if tasks_parquet.exists():
        tasks_df = pd.read_parquet(tasks_parquet)
        with open(tasks_jsonl, 'w') as f:
            for _, row in tasks_df.iterrows():
                task_entry = {
                    "task_index": int(row["task_index"]),
                    "task": row["task"]
                }
                f.write(json.dumps(task_entry) + '\n')
        logger.info(f"Created tasks.jsonl with {len(tasks_df)} tasks for GR00T")

    # Convert episodes/ parquet to episodes.jsonl
    episodes_dir = meta_dir / "episodes"
    episodes_jsonl = meta_dir / "episodes.jsonl"

    if episodes_dir.exists():
        # Find all episode parquet files
        episode_files = sorted(episodes_dir.glob("chunk-*/file-*.parquet"))
        if episode_files:
            episode_dfs = [pd.read_parquet(f) for f in episode_files]
            episodes_df = pd.concat(episode_dfs, ignore_index=True)

            with open(episodes_jsonl, 'w') as f:
                for _, row in episodes_df.iterrows():
                    episode_entry = {
                        "episode_index": int(row["episode_index"]),
                        "tasks": [row["task"]],  # GR00T expects a list of tasks
                        "length": int(row["length"])
                    }
                    f.write(json.dumps(episode_entry) + '\n')
            logger.info(f"Created episodes.jsonl with {len(episodes_df)} episodes for GR00T")

    return tasks_jsonl, episodes_jsonl
