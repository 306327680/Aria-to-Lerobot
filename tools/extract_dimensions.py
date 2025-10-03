#!/usr/bin/env python3
"""
Extract specific dimensions from a LeRobot dataset to create focused subsets.

This tool reads an existing LeRobot dataset and creates a new dataset with only
the specified state and action dimensions, useful for creating multiple experimental
configurations from a single source dataset.

Usage:
    python extract_dimensions.py <source_dataset> <output_dataset> \\
        --state-dims 0:63 --action-dims 3,7 \\
        --state-name "right_hand" --action-name "force_magnitude"
"""

import sys
import json
import argparse
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import logging
from typing import List, Optional, Dict, Any

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def parse_dimension_spec(spec: str) -> List[int]:
    """Parse dimension specification string

    Args:
        spec: Dimension specification, e.g., "0:63", "3,7", "0,5,10:15"

    Returns:
        List of dimension indices
    """
    indices = []
    parts = spec.split(',')

    for part in parts:
        part = part.strip()
        if ':' in part:
            # Range specification
            start, end = part.split(':')
            indices.extend(range(int(start), int(end)))
        else:
            # Single index
            indices.append(int(part))

    return sorted(set(indices))


def extract_parquet_dimensions(
    input_file: Path,
    output_file: Path,
    state_dims: Optional[List[int]],
    action_dims: Optional[List[int]]
):
    """Extract specified dimensions from a parquet file

    Args:
        input_file: Input parquet file path
        output_file: Output parquet file path
        state_dims: List of state dimension indices to extract (None = all)
        action_dims: List of action dimension indices to extract (None = all)
    """
    # Read parquet file
    df = pd.read_parquet(input_file)

    # Extract state dimensions
    if state_dims is not None and 'observation.state' in df.columns:
        state_values = np.vstack(df['observation.state'].values)
        extracted_state = state_values[:, state_dims]
        df['observation.state'] = list(extracted_state)

    # Extract action dimensions
    if action_dims is not None and 'action' in df.columns:
        action_values = np.vstack(df['action'].values)
        extracted_action = action_values[:, action_dims]
        df['action'] = list(extracted_action)

    # Write output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, engine='pyarrow')


def update_modality_json(
    modality_file: Path,
    state_dims: Optional[List[int]],
    action_dims: Optional[List[int]],
    state_name: str = "extracted_state",
    action_name: str = "extracted_action"
):
    """Update modality.json with new dimension ranges

    Args:
        modality_file: Path to modality.json
        state_dims: Extracted state dimensions
        action_dims: Extracted action dimensions
        state_name: Name for the new state modality
        action_name: Name for the new action modality
    """
    with open(modality_file, 'r') as f:
        modality = json.load(f)

    # Preserve video and annotation sections
    video_section = modality.get('video', None)
    annotation_section = modality.get('annotation', None)

    # Update state modality
    if state_dims is not None:
        modality['state'] = {
            state_name: {
                "original_key": "observation.state",
                "start": 0,
                "end": len(state_dims),
                "description": f"Extracted state dimensions: {state_dims}"
            }
        }

    # Update action modality
    if action_dims is not None:
        modality['action'] = {
            action_name: {
                "original_key": "action",
                "start": 0,
                "end": len(action_dims),
                "description": f"Extracted action dimensions: {action_dims}"
            }
        }

    # Restore video and annotation sections
    if video_section is not None:
        modality['video'] = video_section
    if annotation_section is not None:
        modality['annotation'] = annotation_section

    with open(modality_file, 'w') as f:
        json.dump(modality, f, indent=2)


def recompute_stats_json(
    stats_file: Path,
    data_dir: Path,
    state_dims: Optional[List[int]],
    action_dims: Optional[List[int]]
):
    """Recompute stats.json from extracted parquet files

    Args:
        stats_file: Path to stats.json
        data_dir: Path to data directory
        state_dims: Extracted state dimensions (None if not extracted)
        action_dims: Extracted action dimensions (None if not extracted)
    """
    def convert_numpy_types(obj):
        """Convert numpy types to Python native types"""
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

    # Find all parquet files
    parquet_files = sorted(data_dir.glob("chunk-*/episode_*.parquet"))

    if not parquet_files:
        logger.error("No parquet files found for stats recomputation")
        return

    logger.info(f"Recomputing statistics from {len(parquet_files)} parquet files...")

    stats_data = {}

    # Read sample to check columns
    sample_df = pd.read_parquet(parquet_files[0])

    # Compute stats for observation.state
    if 'observation.state' in sample_df.columns:
        logger.info("Computing statistics for observation.state...")
        state_data = []
        for pf in parquet_files:
            df = pd.read_parquet(pf)
            state_values = np.vstack(df['observation.state'].values)
            state_data.append(state_values)

        all_state = np.vstack(state_data)
        logger.info(f"  Shape: {all_state.shape}")

        stats_data['observation.state'] = {
            'mean': convert_numpy_types(np.nanmean(all_state, axis=0)),
            'std': convert_numpy_types(np.nanstd(all_state, axis=0)),
            'min': convert_numpy_types(np.nanmin(all_state, axis=0)),
            'max': convert_numpy_types(np.nanmax(all_state, axis=0)),
            'q01': convert_numpy_types(np.nanpercentile(all_state, 1, axis=0)),
            'q99': convert_numpy_types(np.nanpercentile(all_state, 99, axis=0)),
        }
        logger.info(f"  ✓ Computed statistics")

    # Compute stats for action
    if 'action' in sample_df.columns:
        logger.info("Computing statistics for action...")
        action_data = []
        for pf in parquet_files:
            df = pd.read_parquet(pf)
            action_values = np.vstack(df['action'].values)
            action_data.append(action_values)

        all_action = np.vstack(action_data)
        logger.info(f"  Shape: {all_action.shape}")

        stats_data['action'] = {
            'mean': convert_numpy_types(np.mean(all_action, axis=0)),
            'std': convert_numpy_types(np.std(all_action, axis=0)),
            'min': convert_numpy_types(np.min(all_action, axis=0)),
            'max': convert_numpy_types(np.max(all_action, axis=0)),
            'q01': convert_numpy_types(np.percentile(all_action, 1, axis=0)),
            'q99': convert_numpy_types(np.percentile(all_action, 99, axis=0)),
        }
        logger.info(f"  ✓ Computed statistics")

    # Write stats.json
    with open(stats_file, 'w') as f:
        json.dump(stats_data, f, indent=2)

    logger.info(f"✓ Saved stats.json with {len(stats_data)} modalities")


def update_info_json(
    info_file: Path,
    state_dims: Optional[List[int]],
    action_dims: Optional[List[int]],
    format_type: str = "groot"
):
    """Update info.json feature shapes and path patterns

    Args:
        info_file: Path to info.json
        state_dims: Extracted state dimensions
        action_dims: Extracted action dimensions
        format_type: "groot" for training, "lerobot" for visualization
    """
    with open(info_file, 'r') as f:
        info = json.load(f)

    # Update observation.state shape
    if state_dims is not None and 'observation.state' in info.get('features', {}):
        info['features']['observation.state']['shape'] = [len(state_dims)]

    # Update action shape
    if action_dims is not None and 'action' in info.get('features', {}):
        info['features']['action']['shape'] = [len(action_dims)]

    # Set path patterns based on format type
    if format_type == "groot":
        # GR00T format: episode_chunk and episode_index for training
        if 'data_path' in info:
            info['data_path'] = 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet'
        if 'video_path' in info:
            info['video_path'] = 'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4'
    else:  # lerobot
        # LeRobot format: chunk_index and file_index for visualization
        if 'data_path' in info:
            info['data_path'] = 'data/chunk-{chunk_index:03d}/episode_{file_index:06d}.parquet'
        if 'video_path' in info:
            info['video_path'] = 'videos/chunk-{chunk_index:03d}/{video_key}/episode_{file_index:06d}.mp4'

    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)


def extract_dimensions(
    source_dataset: Path,
    output_dataset: Path,
    state_dims: Optional[List[int]],
    action_dims: Optional[List[int]],
    state_name: str = "extracted_state",
    action_name: str = "extracted_action",
    format_type: str = "groot"
):
    """Extract dimensions from source dataset to create new dataset

    Args:
        source_dataset: Source dataset directory
        output_dataset: Output dataset directory
        state_dims: State dimension indices to extract (None = all)
        action_dims: Action dimension indices to extract (None = all)
        state_name: Name for extracted state modality
        action_name: Name for extracted action modality
        format_type: "groot" for training, "lerobot" for visualization
    """
    if not source_dataset.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_dataset}")

    if output_dataset.exists():
        logger.warning(f"Output dataset exists, removing: {output_dataset}")
        shutil.rmtree(output_dataset)

    output_dataset.mkdir(parents=True, exist_ok=True)

    logger.info(f"Extracting dimensions from {source_dataset} to {output_dataset}")
    if state_dims:
        logger.info(f"  State dimensions: {state_dims} ({len(state_dims)} dims)")
    if action_dims:
        logger.info(f"  Action dimensions: {action_dims} ({len(action_dims)} dims)")

    # Process all parquet files
    data_dir = source_dataset / "data"
    parquet_files = sorted(data_dir.glob("chunk-*/episode_*.parquet"))

    logger.info(f"Processing {len(parquet_files)} parquet files...")
    for pf in parquet_files:
        relative_path = pf.relative_to(data_dir)
        output_file = output_dataset / "data" / relative_path
        extract_parquet_dimensions(pf, output_file, state_dims, action_dims)

    logger.info("✓ Parquet files processed")

    # Copy video files (no dimension changes)
    videos_dir = source_dataset / "videos"
    if videos_dir.exists():
        logger.info("Copying video files...")
        shutil.copytree(videos_dir, output_dataset / "videos", dirs_exist_ok=True)
        logger.info("✓ Videos copied")

    # Copy and update metadata
    meta_dir = source_dataset / "meta"
    output_meta = output_dataset / "meta"
    output_meta.mkdir(parents=True, exist_ok=True)

    logger.info("Updating metadata...")

    # Copy tasks files
    for tasks_file in ["tasks.parquet", "tasks.jsonl"]:
        src = meta_dir / tasks_file
        if src.exists():
            shutil.copy(src, output_meta / tasks_file)

    # Copy episodes directory
    episodes_dir = meta_dir / "episodes"
    if episodes_dir.exists():
        shutil.copytree(episodes_dir, output_meta / "episodes", dirs_exist_ok=True)

    # Copy episodes.jsonl
    episodes_jsonl = meta_dir / "episodes.jsonl"
    if episodes_jsonl.exists():
        shutil.copy(episodes_jsonl, output_meta / "episodes.jsonl")

    # Update info.json
    src_info = meta_dir / "info.json"
    dst_info = output_meta / "info.json"
    if src_info.exists():
        shutil.copy(src_info, dst_info)
        update_info_json(dst_info, state_dims, action_dims, format_type)
        logger.info(f"✓ Updated info.json (format: {format_type})")

    # Recompute stats.json from extracted data
    dst_stats = output_meta / "stats.json"
    recompute_stats_json(dst_stats, output_dataset / "data", state_dims, action_dims)

    # Update modality.json
    src_modality = meta_dir / "modality.json"
    dst_modality = output_meta / "modality.json"
    if src_modality.exists():
        shutil.copy(src_modality, dst_modality)
        update_modality_json(dst_modality, state_dims, action_dims, state_name, action_name)
        logger.info("✓ Updated modality.json")

    logger.info(f"\n✅ Dataset created: {output_dataset}")

    # Print summary
    state_desc = f"{len(state_dims)} dims" if state_dims else "all dims"
    action_desc = f"{len(action_dims)} dims" if action_dims else "all dims"
    logger.info(f"   State: {state_desc}")
    logger.info(f"   Action: {action_desc}")


def main():
    parser = argparse.ArgumentParser(
        description="""
╔════════════════════════════════════════════════════════════════════════════╗
║  Extract Dimensions Tool - Create experimental subsets from full dataset  ║
╚════════════════════════════════════════════════════════════════════════════╝

This tool extracts specific state and action dimensions from a full LeRobot
dataset, useful for creating multiple control group experiments with different
data configurations.
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
═══════════════════════════════════════════════════════════════════════════════
EXAMPLES
═══════════════════════════════════════════════════════════════════════════════

1. Extract right hand + force magnitudes for VISUALIZATION:
   python extract_dimensions.py output_glue output_glue_viz \\
       --state-dims 0:63 --action-dims 0,1 \\
       --state-name "right_hand" --action-name "force_magnitude" \\
       --format lerobot

2. Extract right hand + force magnitudes for TRAINING (default):
   python extract_dimensions.py output_glue output_glue_train \\
       --state-dims 0:63 --action-dims 0,1 \\
       --state-name "right_hand" --action-name "force_magnitude"

3. Extract only fingertips (pinch gesture):
   python extract_dimensions.py output_glue output_glue_pinch \\
       --state-dims 12,13,14,24,25,26 --action-dims 0,1

4. Extract all state + only force magnitudes:
   python extract_dimensions.py output_glue output_glue_force_only \\
       --action-dims 0,1

5. Extract only left hand force (magnitude + xyz):
   python extract_dimensions.py output_glue output_glue_left_force \\
       --action-dims 0,2,3,4

═══════════════════════════════════════════════════════════════════════════════
DIMENSION SPECIFICATION FORMAT
═══════════════════════════════════════════════════════════════════════════════

  Single values:  "3,7,12"         → Extract dimensions 3, 7, and 12
  Ranges:         "0:63"           → Extract dimensions 0 to 62 (inclusive)
  Mixed:          "0,5,10:15,20"   → Extract 0, 5, 10-14, and 20

═══════════════════════════════════════════════════════════════════════════════
ACTION DIMENSION MAPPING (from CSV to action vector)
═══════════════════════════════════════════════════════════════════════════════

  Dim 0: left.force_total    → Ring Tip_magnitude
  Dim 1: right.force_total   → Little Tip_magnitude
  Dim 2: left.force_xyz[0]   → Ring Tip_fx
  Dim 3: left.force_xyz[1]   → Ring Tip_fy
  Dim 4: left.force_xyz[2]   → Ring Tip_fz
  Dim 5: right.force_xyz[0]  → Little Tip_fx
  Dim 6: right.force_xyz[1]  → Little Tip_fy
  Dim 7: right.force_xyz[2]  → Little Tip_fz

═══════════════════════════════════════════════════════════════════════════════
STATE DIMENSION MAPPING
═══════════════════════════════════════════════════════════════════════════════

  Dim 0:63   → Right hand landmarks (21 keypoints × 3 coords)
  Dim 63:79  → Device pose (4×4 matrix, 16 elements)

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMATS
═══════════════════════════════════════════════════════════════════════════════

  --format groot    : For GR00T training (uses episode_chunk/episode_index)
  --format lerobot  : For LeRobot visualization (uses chunk_index/file_index)

═══════════════════════════════════════════════════════════════════════════════
"""
    )

    # Required arguments
    parser.add_argument("source_dataset", type=str, metavar="SOURCE",
                       help="[REQUIRED] Source dataset directory path")
    parser.add_argument("output_dataset", type=str, metavar="OUTPUT",
                       help="[REQUIRED] Output dataset directory path")

    # Optional dimension specifications
    parser.add_argument("--state-dims", type=str, metavar="SPEC",
                       help="[OPTIONAL] State dimensions to extract. Format: '0:63' or '3,7,12'. Omit to keep all.")
    parser.add_argument("--action-dims", type=str, metavar="SPEC",
                       help="[OPTIONAL] Action dimensions to extract. Format: '0,1' or '2:5'. Omit to keep all.")

    # Optional naming
    parser.add_argument("--state-name", type=str, default="extracted_state", metavar="NAME",
                       help="[OPTIONAL] Name for extracted state modality. Default: 'extracted_state'")
    parser.add_argument("--action-name", type=str, default="extracted_action", metavar="NAME",
                       help="[OPTIONAL] Name for extracted action modality. Default: 'extracted_action'")

    # Optional format
    parser.add_argument("--format", type=str, choices=["groot", "lerobot"], default="groot", metavar="FORMAT",
                       help="[OPTIONAL] Output format. Choices: {'groot', 'lerobot'}. Default: 'groot'")

    args = parser.parse_args()

    source_dataset = Path(args.source_dataset)
    output_dataset = Path(args.output_dataset)

    # Parse dimension specifications
    state_dims = parse_dimension_spec(args.state_dims) if args.state_dims else None
    action_dims = parse_dimension_spec(args.action_dims) if args.action_dims else None

    if state_dims is None and action_dims is None:
        logger.error("Must specify at least one of --state-dims or --action-dims")
        return 1

    extract_dimensions(
        source_dataset=source_dataset,
        output_dataset=output_dataset,
        state_dims=state_dims,
        action_dims=action_dims,
        state_name=args.state_name,
        action_name=args.action_name,
        format_type=args.format
    )

    logger.info("\n🎉 Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
