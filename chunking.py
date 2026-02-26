"""
Chunk processing for multi-episode datasets

Handles splitting recordings into episodes based on time boundaries
and processing all episodes in a single pass through the VRS file.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Set
import time

from .io_parquet import ParquetWriter
from .utils import ConversionProgress
from .frame_processing import extract_frame_data as frame_extract, update_processing_stats
from .visualize import create_visualization_video

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
    encode_videos_func,
    create_visualization: bool = False,
    rotate_visualization: bool = False,
    vrs_path: Optional[str] = None,
    mps_dir: Optional[str] = None,
    active_chunk_indices: Optional[Set[int]] = None,
    episode_id_map: Optional[Dict[int, int]] = None
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
        active_chunk_indices: Optional set of chunk indices to process (skip others)
        episode_id_map: Optional mapping from chunk_idx to episode_id for sequential naming

    Returns:
        List of chunk results (one per active chunk/episode)
    """
    num_chunks = len(chunk_boundaries_ns) - 1
    active_count = len(active_chunk_indices) if active_chunk_indices else num_chunks
    logger.info(f"Processing {active_count} active chunks out of {num_chunks} total in single pass through VRS")

    # Initialize writers for all chunks
    chunk_writers = []
    chunk_progresses = []
    chunk_data = []

    for chunk_idx in range(num_chunks):
        # Skip inactive chunks (gaps between segments)
        if active_chunk_indices is not None and chunk_idx not in active_chunk_indices:
            chunk_writers.append(None)
            chunk_progresses.append(None)
            chunk_data.append(None)
            continue

        # Use mapped episode ID for file naming (sequential: 0, 1, 2...)
        episode_id = episode_id_map[chunk_idx] if episode_id_map else chunk_idx

        # Create parquet writer for this chunk/episode
        writer = ParquetWriter(
            output_dir=structure["data"],
            schema=schema,
            base_filename=f"episode_{episode_id:06d}"
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

        # Skip frames routed to inactive chunks (gaps between segments)
        if active_chunk_indices is not None and chunk_idx not in active_chunk_indices:
            continue

        # Get chunk-specific data
        data = chunk_data[chunk_idx]
        writer = chunk_writers[chunk_idx]
        progress = chunk_progresses[chunk_idx]

        # Use mapped episode ID for parquet metadata
        episode_id = episode_id_map[chunk_idx] if episode_id_map else chunk_idx

        # Calculate global index dynamically
        active_indices_before = [i for i in range(chunk_idx)
                                 if active_chunk_indices is None or i in active_chunk_indices]
        global_index = sum(chunk_data[i]["valid_frames"] for i in active_indices_before) + data["valid_frames"]

        # Extract frame data
        chunk_start_ns = chunk_boundaries_ns[chunk_idx]
        frame_data = frame_extract(
            session, timestamp_ns, data["step"], episode_id,
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
        # Skip inactive chunks
        if active_chunk_indices is not None and chunk_idx not in active_chunk_indices:
            continue

        data = chunk_data[chunk_idx]
        writer = chunk_writers[chunk_idx]
        episode_id = episode_id_map[chunk_idx] if episode_id_map else chunk_idx

        global_start = cumulative_frames
        global_end = cumulative_frames + data['valid_frames'] - 1

        logger.info(f"Episode {episode_id} (chunk {chunk_idx}): {data['valid_frames']} frames, "
                   f"global index {global_start}-{global_end}")

        # Finalize parquet writer
        writer.finalize()

        # Write video for this chunk
        if data["video_frames"]:
            encode_videos_func(data["video_frames"], structure["video_main"], fps, episode_id)

        # Generate visualization video for this episode if requested
        if create_visualization and vrs_path and mps_dir:
            logger.info(f"Generating visualization for episode {episode_id}...")
            try:
                # Get output directory from structure
                out_dir = structure["root"]
                viz_output_path = out_dir / f"visualization_episode_{episode_id:06d}.mp4"

                # Calculate time range for this episode
                start_time_s = (chunk_boundaries_ns[chunk_idx] - start_timestamp_ns) / 1e9
                if chunk_boundaries_ns[chunk_idx + 1] is None:
                    # Last chunk - use all remaining frames
                    end_time_s = None
                    max_viz_frames = None
                else:
                    end_time_s = (chunk_boundaries_ns[chunk_idx + 1] - start_timestamp_ns) / 1e9
                    # Estimate frames in this time range
                    duration_s = end_time_s - start_time_s
                    max_viz_frames = int(duration_s * fps) + 10  # Add buffer

                # Calculate start timestamp for visualization
                start_ts_ns = chunk_boundaries_ns[chunk_idx]

                create_visualization_video(
                    vrs_path=vrs_path,
                    mps_dir=mps_dir,
                    output_path=str(viz_output_path),
                    max_frames=max_viz_frames,
                    stride=stride,
                    rotate_image=rotate_visualization,
                    haptic_loader=haptic_loader,
                    start_timestamp_ns=start_ts_ns,
                    global_start_timestamp_ns=start_timestamp_ns,  # Use global start for haptic data alignment
                    session=session  # 🔥 Reuse the same session for all episodes!
                )
                logger.info(f"Visualization for episode {episode_id} created: {viz_output_path}")
            except Exception as e:
                logger.warning(f"Failed to create visualization for episode {episode_id}: {e}")

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
            'chunk_idx': episode_id,
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


def load_segment_pairs(segments_csv: str) -> List[Tuple[float, float]]:
    """Load enabled force segment start/end pairs from alignment tool CSV.

    Reads CSV with columns: segment_id, start_time_s, end_time_s, duration_s, enabled
    Only returns segments where enabled == 'true'.

    Args:
        segments_csv: Path to segments CSV file from Aria-Haptic alignment tool

    Returns:
        Sorted list of (start_time_s, end_time_s) tuples for enabled segments
    """
    import csv
    segments = []

    with open(segments_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            enabled = row.get('enabled', 'true').strip().lower()
            if enabled == 'true':
                start = float(row['start_time_s'])
                end = float(row['end_time_s'])
                segments.append((start, end))

    segments.sort(key=lambda s: s[0])
    logger.info(f"Loaded {len(segments)} enabled segments from {segments_csv}")
    return segments


def create_segment_boundaries(
    segments: List[Tuple[float, float]],
    start_timestamp_ns: int
) -> Tuple[List[int], Set[int], Dict[int, int]]:
    """Create interleaved chunk boundaries from segment pairs.

    Produces boundaries: [seg0_start, seg0_end, seg1_start, seg1_end, ..., None]
    Even-indexed chunks (0, 2, 4...) are active segments.
    Odd-indexed chunks (1, 3, 5...) are gaps to be skipped.

    Args:
        segments: Sorted list of (start_time_s, end_time_s) tuples
        start_timestamp_ns: Global start timestamp in nanoseconds

    Returns:
        Tuple of:
        - boundaries_ns: Interleaved boundary timestamps with None sentinel
        - active_indices: Set of active chunk indices (even numbers)
        - episode_id_map: Mapping from chunk_idx to sequential episode_id
    """
    boundaries_ns = []
    for start_s, end_s in segments:
        boundaries_ns.append(int(start_s * 1e9) + start_timestamp_ns)
        boundaries_ns.append(int(end_s * 1e9) + start_timestamp_ns)
    boundaries_ns.append(None)  # sentinel for last boundary

    # Even chunks = active segments, odd chunks = gaps
    num_chunks = len(boundaries_ns) - 1
    active_indices = set(range(0, num_chunks, 2))

    # Map active chunk indices to sequential episode IDs: {0: 0, 2: 1, 4: 2, ...}
    episode_id_map = {}
    episode_counter = 0
    for idx in sorted(active_indices):
        episode_id_map[idx] = episode_counter
        episode_counter += 1

    logger.info(f"Created {len(segments)} segment boundaries "
                f"({len(active_indices)} active, {num_chunks - len(active_indices)} gaps)")
    return boundaries_ns, active_indices, episode_id_map
