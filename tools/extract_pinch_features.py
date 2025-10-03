#!/usr/bin/env python3
"""
Extract pinch features: thumb tip + index tip positions (6D) + distance (1D) = 7D state.

Extracts:
- Thumb tip (keypoint 4): x, y, z
- Index tip (keypoint 8): x, y, z  
- Pinch distance: Euclidean distance between them
Total: 7 dimensions for state
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


def extract_pinch_features(landmarks: np.ndarray) -> np.ndarray:
    """
    Extract pinch features from hand landmarks.

    Args:
        landmarks: (63,) array of right hand landmarks (21 keypoints × 3 coords)

    Returns:
        (7,) array: [thumb_x, thumb_y, thumb_z, index_x, index_y, index_z, distance]
    """
    if landmarks.shape == (63,):
        landmarks = landmarks.reshape(21, 3)

    # Thumb tip: keypoint 4 → indices 12, 13, 14
    # Index finger tip: keypoint 8 → indices 24, 25, 26
    thumb_tip = landmarks[4]  # (x, y, z)
    index_tip = landmarks[8]  # (x, y, z)

    # Euclidean distance
    distance = np.linalg.norm(thumb_tip - index_tip)

    # Combine: [thumb_xyz, index_xyz, distance]
    features = np.concatenate([thumb_tip, index_tip, [distance]]).astype(np.float32)

    # Replace NaN values with 0
    features = np.nan_to_num(features, nan=0.0)

    return features


def extract_parquet_pinch_features(
    input_file: Path,
    output_file: Path,
    action_dims: Optional[List[int]]
):
    """Extract pinch features from parquet file"""
    df = pd.read_parquet(input_file)
    
    # Extract pinch features from observation.state (right hand landmarks)
    state_values = np.vstack(df['observation.state'].values)
    
    # Compute pinch features for each frame
    pinch_features_list = []
    for i in range(len(state_values)):
        # Extract right hand landmarks (first 63 dims)
        hand_landmarks = state_values[i, :63]
        features = extract_pinch_features(hand_landmarks)
        pinch_features_list.append(features)
    
    # Replace observation.state with pinch features (7D)
    df['observation.state'] = list(pinch_features_list)
    
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
        description="Extract pinch features: thumb/index positions + distance (7D state)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
State features (7 dimensions):
  [0:3] - Thumb tip position (x, y, z)
  [3:6] - Index tip position (x, y, z)
  [6]   - Pinch distance (Euclidean distance)

Example:
  # Extract pinch features + force magnitudes (for visualization)
  python extract_pinch_features.py output_glue output_glue_pinch_viz \\
      --action-dims 0,1 --format lerobot
  
  # Extract pinch features + all actions (for training)
  python extract_pinch_features.py output_glue output_glue_pinch_full \\
      --format groot
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
    
    logger.info(f"Extracting pinch features from {source_dataset}")
    logger.info(f"  State: thumb_xyz + index_xyz + distance (7 dims)")
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
        extract_parquet_pinch_features(pf, output_file, action_dims)
    
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
        
        # Update observation.state shape to 7 (pinch features)
        if 'observation.state' in info.get('features', {}):
            info['features']['observation.state']['shape'] = [7]
        
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
            'mean': all_state.mean(axis=0).tolist(),
            'std': all_state.std(axis=0).tolist(),
            'min': all_state.min(axis=0).tolist(),
            'max': all_state.max(axis=0).tolist(),
            'q01': np.percentile(all_state, 1, axis=0).tolist(),
            'q99': np.percentile(all_state, 99, axis=0).tolist(),
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
    logger.info(f"   Distance: min={stats['observation.state']['min'][6]:.4f}, "
               f"max={stats['observation.state']['max'][6]:.4f}")
    
    # Update modality.json
    src_modality = meta_dir / "modality.json"
    if src_modality.exists():
        # Load source modality to preserve video and annotation sections
        with open(src_modality) as f:
            src_modality_data = json.load(f)

        modality = {
            "state": {
                "pinch_features": {
                    "original_key": "observation.state",
                    "start": 0,
                    "end": 7,
                    "description": "Thumb tip xyz (0:3), Index tip xyz (3:6), Distance (6)"
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
    logger.info(f"   State: pinch_features (7 dims)")
    if action_dims:
        logger.info(f"   Action: {len(action_dims)} dims")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
