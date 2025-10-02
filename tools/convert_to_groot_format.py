#!/usr/bin/env python3
"""
Convert LeRobot v3.0 dataset to GR00T-compatible format

This script converts the Parquet-based metadata files to JSONL format
required by NVIDIA Isaac GR00T.
"""

import json
import argparse
from pathlib import Path
import pandas as pd


def convert_tasks_parquet_to_jsonl(meta_dir: Path) -> Path:
    """Convert tasks.parquet to tasks.jsonl for GR00T"""
    tasks_parquet = meta_dir / "tasks.parquet"
    tasks_jsonl = meta_dir / "tasks.jsonl"

    if not tasks_parquet.exists():
        raise FileNotFoundError(f"tasks.parquet not found: {tasks_parquet}")

    tasks_df = pd.read_parquet(tasks_parquet)

    with open(tasks_jsonl, 'w') as f:
        for _, row in tasks_df.iterrows():
            task_entry = {
                "task_index": int(row["task_index"]),
                "task": row["task"]
            }
            f.write(json.dumps(task_entry) + '\n')

    print(f"✅ Created {tasks_jsonl} with {len(tasks_df)} tasks")
    return tasks_jsonl


def convert_episodes_parquet_to_jsonl(meta_dir: Path) -> Path:
    """Convert episodes/ parquet directory to episodes.jsonl for GR00T"""
    episodes_dir = meta_dir / "episodes"
    episodes_jsonl = meta_dir / "episodes.jsonl"

    if not episodes_dir.exists():
        raise FileNotFoundError(f"episodes directory not found: {episodes_dir}")

    # Find all episode parquet files
    episode_files = sorted(episodes_dir.glob("chunk-*/file-*.parquet"))

    if not episode_files:
        raise FileNotFoundError(f"No episode parquet files found in {episodes_dir}")

    # Concatenate all episode dataframes
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

    print(f"✅ Created {episodes_jsonl} with {len(episodes_df)} episodes")
    return episodes_jsonl


def verify_groot_compatibility(dataset_dir: Path):
    """Verify that the dataset has all required GR00T files"""
    meta_dir = dataset_dir / "meta"

    required_files = {
        "modality.json": meta_dir / "modality.json",
        "episodes.jsonl": meta_dir / "episodes.jsonl",
        "tasks.jsonl": meta_dir / "tasks.jsonl",
        "info.json": meta_dir / "info.json",
        "stats.json": meta_dir / "stats.json",
    }

    missing = []
    for name, path in required_files.items():
        if not path.exists():
            missing.append(name)
        else:
            print(f"✓ {name}")

    if missing:
        print(f"\n⚠️  Missing files: {', '.join(missing)}")
        return False

    print(f"\n✅ All required GR00T metadata files present!")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert LeRobot v3.0 dataset to GR00T-compatible format"
    )
    parser.add_argument(
        "dataset_dir",
        type=str,
        help="Path to the LeRobot dataset directory (e.g., output_glue)"
    )

    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)

    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        return 1

    meta_dir = dataset_dir / "meta"
    if not meta_dir.exists():
        print(f"❌ Meta directory not found: {meta_dir}")
        return 1

    print(f"📁 Converting dataset: {dataset_dir}")
    print()

    try:
        # Convert Parquet files to JSONL
        convert_tasks_parquet_to_jsonl(meta_dir)
        convert_episodes_parquet_to_jsonl(meta_dir)

        print()
        print("=" * 60)
        print("Verifying GR00T compatibility...")
        print("=" * 60)

        verify_groot_compatibility(dataset_dir)

        print()
        print("🎉 Conversion complete!")
        print(f"   Dataset is now compatible with NVIDIA Isaac GR00T")
        print(f"   You can load it using:")
        print(f"     LeRobotSingleDataset(dataset_path='{dataset_dir}')")

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
