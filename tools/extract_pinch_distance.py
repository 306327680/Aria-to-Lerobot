#!/usr/bin/env python3
"""
Extract pinch distance (thumb-index distance) as state feature.

Computes Euclidean distance between thumb tip and index finger tip
from the 21-point hand landmarks.
"""

import sys
import json
import argparse
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import logging
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def compute_pinch_distance(landmarks: np.ndarray) -> float:
    """
    Compute distance between thumb tip and index finger tip.
    
    Args:
        landmarks: (63,) array of right hand landmarks (21 keypoints × 3 coords)
                   or (21, 3) array
    
    Returns:
        Euclidean distance between thumb tip and index finger tip
    """
    if landmarks.shape == (63,):
        landmarks = landmarks.reshape(21, 3)
    
    # Thumb tip: keypoint 4 → indices 12, 13, 14
    # Index finger tip: keypoint 8 → indices 24, 25, 26
    thumb_tip = landmarks[4]  # (x, y, z)
    index_tip = landmarks[8]  # (x, y, z)
    
    # Euclidean distance
    distance = np.linalg.norm(thumb_tip - index_tip)
    
    return float(distance)


def extract_parquet_pinch_distance(
    input_file: Path,
    output_file: Path,
    action_dims: Optional[List[int]]
):
    """Extract pinch distance from parquet file"""
    df = pd.read_parquet(input_file)
    
    # Extract pinch distance from observation.state (right hand landmarks)
    state_values = np.vstack(df['observation.state'].values)
    
    # Compute pinch distance for each frame
    pinch_distances = []
    for i in range(len(state_values)):
        # Extract right hand landmarks (first 63 dims)
        hand_landmarks = state_values[i, :63]
        distance = compute_pinch_distance(hand_landmarks)
        pinch_distances.append(distance)

    # Replace observation.state with pinch distance as numpy array (1D)
    df['observation.state'] = [np.array([d], dtype=np.float32) for d in pinch_distances]
    
    # Extract action dimensions if specified
    if action_dims is not None and 'action' in df.columns:
        action_values = np.vstack(df['action'].values)
        extracted_action = action_values[:, action_dims]
        df['action'] = list(extracted_action)
    
    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, engine='pyarrow')


def main():
    parser = argparse.ArgumentParser(
        description="Extract pinch distance (thumb-index) as state feature",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  # Extract pinch distance + force magnitudes
  python extract_pinch_distance.py output_glue output_glue_pinch_distance \\
      --action-dims 0,1 --format groot
  
  # Extract pinch distance + all actions
  python extract_pinch_distance.py output_glue output_glue_pinch_full \\
      --format lerobot
"""
    )
    
    parser.add_argument("source_dataset", type=str, metavar="SOURCE",
                       help="Source dataset directory")
    parser.add_argument("output_dataset", type=str, metavar="OUTPUT",
                       help="Output dataset directory")
    
    parser.add_argument("--action-dims", type=str, metavar="SPEC",
                       help="Action dimensions to extract (e.g., '0,1'). Omit to keep all.")
    
    parser.add_argument("--format", type=str, choices=["groot", "lerobot"], default="groot",
                       help="Output format: 'groot' (training) or 'lerobot' (visualization)")
    
    args = parser.parse_args()
    
    source_dataset = Path(args.source_dataset)
    output_dataset = Path(args.output_dataset)
    
    if not source_dataset.exists():
        logger.error(f"Source dataset not found: {source_dataset}")
        return 1
    
    if output_dataset.exists():
        logger.warning(f"Output dataset exists, removing: {output_dataset}")
        shutil.rmtree(output_dataset)
    
    output_dataset.mkdir(parents=True)
    
    # Parse action dimensions
    action_dims = None
    if args.action_dims:
        parts = args.action_dims.split(',')
        action_dims = [int(p.strip()) for p in parts]
    
    logger.info(f"Extracting pinch distance from {source_dataset}")
    logger.info(f"  State: pinch_distance (1 dim)")
    if action_dims:
        logger.info(f"  Action: {action_dims} ({len(action_dims)} dims)")
    else:
        logger.info(f"  Action: all dimensions")
    
    # Process parquet files
    data_dir = source_dataset / "data"
    parquet_files = sorted(data_dir.glob("chunk-*/episode_*.parquet"))
    
    logger.info(f"Processing {len(parquet_files)} parquet files...")
    for pf in parquet_files:
        relative_path = pf.relative_to(data_dir)
        output_file = output_dataset / "data" / relative_path
        extract_parquet_pinch_distance(pf, output_file, action_dims)
    
    logger.info("✓ Parquet files processed")
    
    # Copy videos
    videos_dir = source_dataset / "videos"
    if videos_dir.exists():
        logger.info("Copying videos...")
        shutil.copytree(videos_dir, output_dataset / "videos", dirs_exist_ok=True)
        logger.info("✓ Videos copied")
    
    # Copy and update metadata
    meta_dir = source_dataset / "meta"
    output_meta = output_dataset / "meta"
    output_meta.mkdir(parents=True, exist_ok=True)
    
    # Copy tasks files
    for tasks_file in ["tasks.parquet", "tasks.jsonl"]:
        src = meta_dir / tasks_file
        if src.exists():
            shutil.copy(src, output_meta / tasks_file)
    
    # Copy episodes
    episodes_dir = meta_dir / "episodes"
    if episodes_dir.exists():
        shutil.copytree(episodes_dir, output_meta / "episodes", dirs_exist_ok=True)
    
    episodes_jsonl = meta_dir / "episodes.jsonl"
    if episodes_jsonl.exists():
        shutil.copy(episodes_jsonl, output_meta / "episodes.jsonl")
    
    # Update info.json
    src_info = meta_dir / "info.json"
    dst_info = output_meta / "info.json"
    if src_info.exists():
        with open(src_info) as f:
            info = json.load(f)
        
        # Update observation.state shape to 1 (pinch distance)
        if 'observation.state' in info.get('features', {}):
            info['features']['observation.state']['shape'] = [1]
        
        # Update action shape
        if action_dims is not None and 'action' in info.get('features', {}):
            info['features']['action']['shape'] = [len(action_dims)]
        
        # Set path format
        if args.format == "groot":
            info['data_path'] = 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet'
            info['video_path'] = 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4'
        else:
            info['data_path'] = 'data/chunk-{chunk_index:03d}/episode_{file_index:06d}.parquet'
            info['video_path'] = 'videos/chunk-{chunk_index:03d}/{video_key}/episode_{file_index:06d}.mp4'
        
        with open(dst_info, 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"✓ Updated info.json (format: {args.format})")
    
    # Recompute stats.json
    logger.info("Recomputing statistics...")
    
    # Collect all state and action data
    state_data = []
    action_data = []
    
    for pf in sorted((output_dataset / "data").glob("chunk-*/episode_*.parquet")):
        df = pd.read_parquet(pf)
        state_values = np.vstack(df['observation.state'].values)
        state_data.append(state_values)
        
        if 'action' in df.columns:
            action_values = np.vstack(df['action'].values)
            action_data.append(action_values)
    
    all_state = np.vstack(state_data)
    all_action = np.vstack(action_data) if action_data else None
    
    stats = {
        'observation.state': {
            'mean': float(np.nanmean(all_state)),
            'std': float(np.nanstd(all_state)),
            'min': float(np.nanmin(all_state)),
            'max': float(np.nanmax(all_state)),
            'q01': float(np.nanpercentile(all_state, 1)),
            'q99': float(np.nanpercentile(all_state, 99)),
        }
    }
    
    if all_action is not None:
        stats['action'] = {
            'mean': all_action.mean(axis=0).tolist(),
            'std': all_action.std(axis=0).tolist(),
            'min': all_action.min(axis=0).tolist(),
            'max': all_action.max(axis=0).tolist(),
            'q01': np.percentile(all_action, 1, axis=0).tolist(),
            'q99': np.percentile(all_action, 99, axis=0).tolist(),
        }
    
    with open(output_meta / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info("✓ Stats computed")
    logger.info(f"   Pinch distance: min={stats['observation.state']['min']:.4f}, "
               f"max={stats['observation.state']['max']:.4f}, "
               f"mean={stats['observation.state']['mean']:.4f}")
    
    # Update modality.json
    src_modality = meta_dir / "modality.json"
    if src_modality.exists():
        # Load source modality to preserve video and annotation sections
        with open(src_modality) as f:
            src_modality_data = json.load(f)

        modality = {
            "state": {
                "pinch_distance": {
                    "original_key": "observation.state",
                    "start": 0,
                    "end": 1,
                    "description": "Euclidean distance between thumb tip and index finger tip"
                }
            }
        }

        if action_dims is not None:
            modality["action"] = {
                "force_magnitude": {
                    "original_key": "action",
                    "start": 0,
                    "end": len(action_dims),
                    "description": f"Extracted action dimensions: {action_dims}"
                }
            }

        # Preserve video and annotation from source
        if "video" in src_modality_data:
            modality["video"] = src_modality_data["video"]
        if "annotation" in src_modality_data:
            modality["annotation"] = src_modality_data["annotation"]

        with open(output_meta / "modality.json", 'w') as f:
            json.dump(modality, f, indent=2)

        logger.info("✓ Updated modality.json")
    
    logger.info(f"\n✅ Dataset created: {output_dataset}")
    logger.info(f"   State: pinch_distance (1 dim)")
    if action_dims:
        logger.info(f"   Action: {len(action_dims)} dims")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
