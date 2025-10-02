"""
Chunk processing for multi-episode datasets

Handles splitting recordings into episodes based on time boundaries
and processing all episodes in a single pass through the VRS file.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import time

from .io_parquet import ParquetWriter
from .utils import ConversionProgress
from .frame_processing import extract_frame_data as frame_extract, update_processing_stats

logger = logging.getLogger(__name__)


def process_all_chunks_single_pass(
    session,
    chunk_boundaries_ns: List[int],
    start_timestamp_ns: int,
    structure: Dict,
    schema,
    max_slop_ns: int,
    fps: float,
    stride: int,
    max_frames: Optional[int],
    haptic_loader: Optional[Any],
    legacy_v21: bool,
    create_camera_frame_iterator,
    encode_videos_func
) -> List[Dict]:
    """
    Process all chunks in a SINGLE PASS through VRS file.
    Avoids re-reading the VRS file multiple times.

    Args:
        session: AriaSession instance
        chunk_boundaries_ns: List of chunk boundary timestamps [start0, start1, ..., startN, None]
        start_timestamp_ns: Global dataset start timestamp
        structure: Output directory structure
        schema: LeRobot schema
        max_slop_ns: Maximum time alignment tolerance
        fps: Frames per second
        stride: Frame stride
        max_frames: Maximum frames to process (optional)
        haptic_loader: Optional HapticDataLoader instance
        legacy_v21: Use LeRobot v2.1 format
        create_camera_frame_iterator: Function to create frame iterator
        encode_videos_func: Function to encode videos

    Returns:
        List of chunk results (one per chunk/episode)
    """
    num_chunks = len(chunk_boundaries_ns) - 1
    logger.info(f"Processing {num_chunks} chunks in single pass through VRS")

    # Initialize writers for all chunks
    chunk_writers = []
    chunk_progresses = []
    chunk_data = []

    for chunk_idx in range(num_chunks):
        # Create parquet writer for this chunk/episode
        writer = ParquetWriter(
            output_dir=structure["data"],
            schema=schema,
            base_filename=f"episode_{chunk_idx:06d}"
        )
        chunk_writers.append(writer)

        # Create progress tracker for this chunk
        progress = ConversionProgress(1000)
        chunk_progresses.append(progress)

        # Initialize data collection for this chunk
        chunk_data.append({
            "valid_frames": 0,
            "dropped_frames": 0,
            "video_frames": [],
            "hand_stats": {"valid": 0, "total": 0},
            "pose_stats": {"valid": 0, "total": 0},
            "nan_rates": {},
            "step": 0,  # Frame index within this chunk
        })

    # Create SINGLE frame iterator for entire VRS
    frame_iterator = create_camera_frame_iterator(session, stride, max_frames)

    # Process all frames ONCE, routing each to appropriate chunk
    for timestamp_ns, image_array in frame_iterator:
        # Determine which chunk this frame belongs to
        chunk_idx = None
        for i in range(num_chunks):
            start_ns = chunk_boundaries_ns[i]
            end_ns = chunk_boundaries_ns[i + 1]

            if end_ns is None:  # Last chunk
                if timestamp_ns >= start_ns:
                    chunk_idx = i
                    break
            else:
                if start_ns <= timestamp_ns < end_ns:
                    chunk_idx = i
                    break

        if chunk_idx is None:
            continue  # Frame outside all chunks

        # Get chunk-specific data
        data = chunk_data[chunk_idx]
        writer = chunk_writers[chunk_idx]
        progress = chunk_progresses[chunk_idx]

        # Calculate global index dynamically
        global_index = sum(chunk_data[i]["valid_frames"] for i in range(chunk_idx)) + data["valid_frames"]

        # Extract frame data
        chunk_start_ns = chunk_boundaries_ns[chunk_idx]
        frame_data = frame_extract(
            session, timestamp_ns, data["step"], chunk_idx,
            max_slop_ns, chunk_start_ns, haptic_loader,
            global_index, fps, start_timestamp_ns  # Pass global start timestamp for haptic alignment
        )

        if frame_data is not None:
            # Write to parquet
            writer.add_row(frame_data)

            # Update stats
            data["valid_frames"] += 1
            data["step"] += 1
            # Use relative timestamp (same as in parquet) for video frames
            relative_timestamp_s = frame_data["timestamp"]
            data["video_frames"].append((relative_timestamp_s, image_array))

            # Update hand/pose stats
            update_processing_stats(frame_data, data)

            progress.update(1)

    # Finalize all chunks
    logger.info("Finalizing all chunks...")
    chunk_results = []
    cumulative_frames = 0

    for chunk_idx in range(num_chunks):
        data = chunk_data[chunk_idx]
        writer = chunk_writers[chunk_idx]

        global_start = cumulative_frames
        global_end = cumulative_frames + data['valid_frames'] - 1

        logger.info(f"Chunk {chunk_idx}: {data['valid_frames']} frames, "
                   f"global index {global_start}-{global_end}")

        # Finalize parquet writer
        writer.finalize()

        # Write video for this chunk
        if data["video_frames"]:
            encode_videos_func(data["video_frames"], structure["video_main"], fps, chunk_idx)

        # Calculate statistics
        if data["hand_stats"]["total"] > 0:
            data["hand_stats"]["valid_rate"] = (
                data["hand_stats"]["valid"] / data["hand_stats"]["total"]
            )
        else:
            data["hand_stats"]["valid_rate"] = 0.0

        if data["pose_stats"]["total"] > 0:
            data["pose_stats"]["valid_rate"] = (
                data["pose_stats"]["valid"] / data["pose_stats"]["total"]
            )
        else:
            data["pose_stats"]["valid_rate"] = 0.0

        # Prepare result
        chunk_results.append({
            'chunk_idx': chunk_idx,
            'start_time_s': (chunk_boundaries_ns[chunk_idx] - start_timestamp_ns) / 1e9,
            'end_time_s': 'end' if chunk_boundaries_ns[chunk_idx+1] is None else (chunk_boundaries_ns[chunk_idx+1] - start_timestamp_ns) / 1e9,
            'result': {
                'success': True,
                'processing': data,
                'files': {
                    'parquet_shards': 1,
                    'video_shards': 1,
                },
            }
        })

        cumulative_frames += data['valid_frames']

    return chunk_results


def load_split_points(split_csv: str) -> List[float]:
    """Load split points from CSV file

    Args:
        split_csv: Path to CSV file with split_time_s column

    Returns:
        List of split times in seconds
    """
    import csv
    split_times_s = []

    with open(split_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            split_time = float(row['split_time_s'])
            split_times_s.append(split_time)

    logger.info(f"Loaded {len(split_times_s)} split points from {split_csv}")
    return split_times_s


def create_chunk_boundaries(
    split_times_s: List[float],
    start_timestamp_ns: int
) -> List[int]:
    """Create chunk boundaries in nanoseconds

    Args:
        split_times_s: List of split times in seconds
        start_timestamp_ns: Start timestamp in nanoseconds

    Returns:
        List of boundary timestamps [start0, start1, ..., startN, None]
    """
    split_timestamps_ns = [
        int(split_time_s * 1e9) + start_timestamp_ns
        for split_time_s in split_times_s
    ]
    chunk_boundaries_ns = [start_timestamp_ns] + split_timestamps_ns + [None]

    logger.info(f"Created {len(chunk_boundaries_ns) - 1} chunk boundaries")
    return chunk_boundaries_ns
