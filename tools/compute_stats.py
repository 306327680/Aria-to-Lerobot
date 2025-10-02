#!/usr/bin/env python3
"""
Compute statistics for existing LeRobot dataset

This script computes mean, std, min, max for observation.state and action columns
and updates stats.json. Required for GR00T compatibility.
"""

import sys
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


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


def compute_stats(dataset_path: Path):
    """Compute statistics for dataset"""
    data_dir = dataset_path / "data"
    meta_dir = dataset_path / "meta"
    stats_file = meta_dir / "stats.json"

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        return False

    # Find all parquet files
    parquet_files = sorted(data_dir.glob("chunk-*/episode_*.parquet"))

    if not parquet_files:
        logger.error("No parquet files found")
        return False

    logger.info(f"Computing statistics from {len(parquet_files)} parquet files...")

    # Read first file to get column structure
    sample_df = pd.read_parquet(parquet_files[0])
    logger.info(f"Columns: {list(sample_df.columns)}")

    stats_data = {}

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

        # Compute mean and std, handling NaN values
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

    logger.info(f"\n✅ Statistics saved to {stats_file}")
    logger.info(f"   Modalities: {list(stats_data.keys())}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Compute statistics for LeRobot dataset")
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the dataset directory (e.g., output_glue)"
    )

    args = parser.parse_args()
    dataset_path = Path(args.dataset_path)

    if not dataset_path.exists():
        logger.error(f"Dataset not found: {dataset_path}")
        return 1

    logger.info(f"📊 Computing statistics for: {dataset_path}\n")

    if compute_stats(dataset_path):
        logger.info("\n🎉 Done!")
        return 0
    else:
        logger.error("\n❌ Failed to compute statistics")
        return 1


if __name__ == "__main__":
    sys.exit(main())
