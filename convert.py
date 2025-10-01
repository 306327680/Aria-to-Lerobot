"""
Core conversion logic from Aria to LeRobot format

Main conversion function that orchestrates the entire process:
- Load Aria data via aria_api
- Iterate over camera-rgb frames as primary timeline
- Extract aligned multi-modal data (pose, hands, etc.)
- Write to LeRobot v3.0 or v2.1 format
"""

import logging
import sys
import time
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple, Iterator
import numpy as np
from dataclasses import asdict

# Add aria_api to path
sys.path.insert(0, str(Path(__file__).parent.parent / "aria_api"))

from aria_api import AriaSession

from .schema import (
    LeRobotSchema, LEROBOT_V3_SCHEMA, LEROBOT_V21_SCHEMA,
    flatten_hand_landmarks, create_tactile_placeholders, flatten_pose_matrix
)
from .video import encode_video_shards, ensure_rgb_format
from .io_parquet import ParquetWriter, write_episode_metadata
from .utils import (
    setup_logging, validate_paths, create_output_structure,
    estimate_fps_from_timestamps, ConversionProgress, validate_conversion_output
)
from .visualize import create_visualization_video
from .metadata import update_consolidated_metadata, write_single_episode_metadata
from .frame_processing import process_frames
from .chunking import (
    process_all_chunks_single_pass, load_split_points, create_chunk_boundaries
)

# Haptic data support
import pandas as pd


logger = logging.getLogger(__name__)


class HapticDataLoader:
    """触觉数据加载器 - Loads and interpolates haptic force sensor data"""

    def __init__(self, csv_path: str, time_offset: float = 0.0):
        """
        Initialize haptic data loader

        Args:
            csv_path: Path to haptic CSV file
            time_offset: Time offset in seconds (haptic data needs to be delayed by this amount)
        """
        self.csv_path = csv_path
        self.time_offset = time_offset

        # Load CSV data
        df = pd.read_csv(csv_path)

        # Extract timestamps (use relative_time column)
        if 'relative_time' in df.columns:
            self.timestamps = df['relative_time'].values
        elif 'timestamp' in df.columns:
            timestamps = df['timestamp'].values
            # Convert to relative time if absolute timestamp
            if timestamps[0] > 1e9:
                timestamps = timestamps / 1e9
            self.timestamps = timestamps - timestamps[0]
        else:
            raise ValueError("CSV file must have 'relative_time' or 'timestamp' column")

        # Extract force data (magnitude + xyz components)
        magnitude_cols = [col for col in df.columns if 'magnitude' in col.lower()]

        if len(magnitude_cols) >= 2:
            # Assume first two magnitude columns are the two sensors
            self.sensor1_name = magnitude_cols[0].replace('_magnitude', '')
            self.sensor2_name = magnitude_cols[1].replace('_magnitude', '')

            # Extract magnitude
            self.force1_magnitude = df[magnitude_cols[0]].values
            self.force2_magnitude = df[magnitude_cols[1]].values

            # Extract xyz components if available
            fx1_col = f"{self.sensor1_name}_fx"
            fy1_col = f"{self.sensor1_name}_fy"
            fz1_col = f"{self.sensor1_name}_fz"
            fx2_col = f"{self.sensor2_name}_fx"
            fy2_col = f"{self.sensor2_name}_fy"
            fz2_col = f"{self.sensor2_name}_fz"

            if all(col in df.columns for col in [fx1_col, fy1_col, fz1_col]):
                self.force1_xyz = np.stack([df[fx1_col].values, df[fy1_col].values, df[fz1_col].values], axis=1)
            else:
                # Fallback: put magnitude in x, zero in y/z
                self.force1_xyz = np.column_stack([self.force1_magnitude, np.zeros_like(self.force1_magnitude), np.zeros_like(self.force1_magnitude)])

            if all(col in df.columns for col in [fx2_col, fy2_col, fz2_col]):
                self.force2_xyz = np.stack([df[fx2_col].values, df[fy2_col].values, df[fz2_col].values], axis=1)
            else:
                # Fallback: put magnitude in x, zero in y/z
                self.force2_xyz = np.column_stack([self.force2_magnitude, np.zeros_like(self.force2_magnitude), np.zeros_like(self.force2_magnitude)])

        elif len(magnitude_cols) == 1:
            self.sensor1_name = magnitude_cols[0].replace('_magnitude', '')
            self.sensor2_name = "sensor2"
            self.force1_magnitude = df[magnitude_cols[0]].values
            self.force2_magnitude = np.zeros_like(self.force1_magnitude)
            self.force1_xyz = np.column_stack([self.force1_magnitude, np.zeros_like(self.force1_magnitude), np.zeros_like(self.force1_magnitude)])
            self.force2_xyz = np.zeros((len(self.force1_magnitude), 3))
        else:
            raise ValueError(f"Cannot find force magnitude columns in {csv_path}")

        logger.info(f"触觉数据加载完成:")
        logger.info(f"  传感器1: {self.sensor1_name}")
        logger.info(f"  传感器2: {self.sensor2_name}")
        logger.info(f"  样本数: {len(self.timestamps)}")
        logger.info(f"  时间范围: {self.timestamps[0]:.2f}s - {self.timestamps[-1]:.2f}s")
        logger.info(f"  时间偏移: {time_offset:.3f}s")

    def get_force_at_time(self, time_s: float) -> Tuple[Tuple[float, np.ndarray], Tuple[float, np.ndarray]]:
        """
        Get force values at specified time (with linear interpolation)

        Args:
            time_s: Timestamp in seconds

        Returns:
            ((force1_magnitude, force1_xyz), (force2_magnitude, force2_xyz))
            where xyz is a 3-element numpy array [fx, fy, fz]
        """
        # Apply time offset
        query_time = time_s - self.time_offset

        # Boundary checks
        if query_time < self.timestamps[0]:
            # Before haptic data starts, return 0
            return (0.0, np.zeros(3)), (0.0, np.zeros(3))
        elif query_time > self.timestamps[-1]:
            # Out of range, return last value
            return (float(self.force1_magnitude[-1]), self.force1_xyz[-1]), \
                   (float(self.force2_magnitude[-1]), self.force2_xyz[-1])

        # Linear interpolation
        idx = np.searchsorted(self.timestamps, query_time)
        if idx == 0:
            return (float(self.force1_magnitude[0]), self.force1_xyz[0]), \
                   (float(self.force2_magnitude[0]), self.force2_xyz[0])

        # Interpolate
        t0, t1 = self.timestamps[idx-1], self.timestamps[idx]
        f1_mag_0, f1_mag_1 = self.force1_magnitude[idx-1], self.force1_magnitude[idx]
        f2_mag_0, f2_mag_1 = self.force2_magnitude[idx-1], self.force2_magnitude[idx]
        f1_xyz_0, f1_xyz_1 = self.force1_xyz[idx-1], self.force1_xyz[idx]
        f2_xyz_0, f2_xyz_1 = self.force2_xyz[idx-1], self.force2_xyz[idx]

        alpha = (query_time - t0) / (t1 - t0)
        force1_magnitude = f1_mag_0 + alpha * (f1_mag_1 - f1_mag_0)
        force2_magnitude = f2_mag_0 + alpha * (f2_mag_1 - f2_mag_0)
        force1_xyz = f1_xyz_0 + alpha * (f1_xyz_1 - f1_xyz_0)
        force2_xyz = f2_xyz_0 + alpha * (f2_xyz_1 - f2_xyz_0)

        return (float(force1_magnitude), force1_xyz), (float(force2_magnitude), force2_xyz)


def convert(vrs_path: str,
           mps_dir: str,
           out_dir: str,
           max_slop_ns: int = 5_000_000,
           fps: Optional[float] = None,
           episode_splits: Optional[List[int]] = None,
           legacy_v21: bool = False,
           overwrite: bool = False,
           stride: int = 1,
           max_frames: Optional[int] = None,
           log_level: str = "INFO",
           create_visualization: bool = False,
           rotate_visualization: bool = False,
           haptic_csv: Optional[str] = None,
           haptic_time_offset: float = 0.0,
           split_csv: Optional[str] = None,
           time_range_ns: Optional[tuple] = None,
           chunk_id: Optional[int] = None,
           global_index_offset: int = 0) -> Dict[str, Any]:
    """
    Convert Aria VRS/MPS data to LeRobot dataset format

    Args:
        vrs_path: Path to VRS file
        mps_dir: Path to MPS directory
        out_dir: Output directory for dataset
        max_slop_ns: Maximum time alignment tolerance (nanoseconds)
        fps: Target FPS (auto-detect if None)
        episode_splits: Frame indices to split episodes (None for single episode)
        legacy_v21: Use LeRobot v2.1 format instead of v3.0
        overwrite: Overwrite existing output directory
        stride: Frame stride (1 = every frame, 2 = every other frame, etc.)
        max_frames: Maximum frames to process (None for all)
        log_level: Logging level
        create_visualization: Create visualization video with hand landmarks and gaze projections
        rotate_visualization: Rotate visualization video 90 degrees clockwise
        haptic_csv: Path to haptic sensor CSV file
        haptic_time_offset: Time offset (seconds) to align haptic data with Aria data
        split_csv: Path to segment split points CSV (from aria_haptic_align.py)
        time_range_ns: Optional (start_ns, end_ns) tuple to filter frames by timestamp
        chunk_id: Optional chunk ID for output naming

    Returns:
        Conversion metadata dictionary
    """
    
    # Setup logging
    log_file = Path(out_dir) / "conversion.log"
    setup_logging(log_level, log_file)
    
    logger.info("=== Aria to LeRobot Conversion Started ===")
    logger.info(f"VRS file: {vrs_path}")
    logger.info(f"MPS directory: {mps_dir}")
    logger.info(f"Output directory: {out_dir}")
    logger.info(f"Format: LeRobot v{'2.1' if legacy_v21 else '3.0'}")
    
    # Validate inputs
    vrs_path = Path(vrs_path)
    
    # Import AriaSession
    from aria_api.session import AriaSession
    
    # Auto-detect MPS directory if not provided
    if mps_dir is None:
        # Try to auto-detect MPS directory
        detected_mps = AriaSession._auto_detect_mps_dir(vrs_path)
        if detected_mps is None:
            raise ValueError(f"Cannot auto-detect MPS directory for {vrs_path}. Please specify mps_dir parameter.")
        mps_dir = detected_mps
        logger.info(f"Auto-detected MPS directory: {mps_dir}")
    else:
        mps_dir = Path(mps_dir)
    
    out_dir = Path(out_dir)

    # Skip validation if this is a chunk (chunk_id > 0) and directory already exists
    # This allows multiple chunks to be written to the same directory
    skip_validation = chunk_id is not None and chunk_id > 0 and out_dir.exists()

    if not skip_validation:
        validation_errors = validate_paths(vrs_path, mps_dir, out_dir, overwrite)
        if validation_errors:
            for error in validation_errors:
                logger.error(error)
            raise ValueError(f"Validation failed: {validation_errors}")
    
    # Create output structure
    structure = create_output_structure(out_dir, legacy_v21)
    schema = LEROBOT_V21_SCHEMA if legacy_v21 else LEROBOT_V3_SCHEMA
    
    start_time = time.time()
    # Use chunk_id as episode_id if provided, otherwise default to 0
    episode_id = chunk_id if chunk_id is not None else 0

    try:
        # Handle split CSV FIRST - if provided, recursively call convert() for each chunk
        # This needs to happen before session initialization to avoid creating unnecessary sessions
        if split_csv is not None:
            # Initialize temporary session just for FPS estimation
            temp_session = AriaSession.from_paths(vrs_path, mps_dir)
            session_info = temp_session.get_session_info()

            # Estimate FPS if not provided
            if fps is None:
                fps = _estimate_fps(temp_session)
                fps = fps / stride  # Adjust for stride
                logger.info(f"Auto-detected FPS: {fps:.2f} (adjusted for stride={stride})")

            logger.info(f"Loading split points from {split_csv}")
            import csv
            split_times_s = []
            with open(split_csv, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    split_time = float(row['split_time_s'])
                    split_times_s.append(split_time)
            logger.info(f"Loaded {len(split_times_s)} split points, creating {len(split_times_s) + 1} chunks")

            # Get start timestamp
            first_frame = next(iter(_create_camera_frame_iterator(temp_session, 1, 1)))
            start_timestamp_ns = first_frame[0]

            # Create chunks using frame time boundaries
            # All chunks will be in the same output directory with different episode IDs
            # Convert split times to timestamps (ns)
            chunk_results = []
            split_timestamps_ns = [int(split_time_s * 1e9) + start_timestamp_ns for split_time_s in split_times_s]
            chunk_boundaries_ns = [start_timestamp_ns] + split_timestamps_ns + [None]

            logger.info(f"\n{'='*60}")
            logger.info(f"Processing {len(chunk_boundaries_ns) - 1} chunks in SINGLE PASS (no VRS re-reading)")
            logger.info(f"Output directory: {out_dir}")
            logger.info(f"{'='*60}\n")

            # Initialize Aria session ONCE for all chunks
            logger.info("Initializing Aria session for single-pass processing...")
            session = AriaSession.from_paths(vrs_path, mps_dir)
            session_info = session.get_session_info()
            logger.info(f"Session capabilities: {session_info['capabilities']}")

            # Initialize haptic data loader if provided (shared across all chunks)
            haptic_loader = None
            if haptic_csv is not None:
                logger.info(f"Loading haptic data from {haptic_csv} with offset {haptic_time_offset}s")
                haptic_loader = HapticDataLoader(haptic_csv, haptic_time_offset)

            # Process all chunks in ONE PASS through VRS
            chunk_results = process_all_chunks_single_pass(
                session=session,
                chunk_boundaries_ns=chunk_boundaries_ns,
                start_timestamp_ns=start_timestamp_ns,
                structure=structure,
                schema=schema,
                max_slop_ns=max_slop_ns,
                fps=fps,
                stride=stride,
                max_frames=max_frames,
                haptic_loader=haptic_loader,
                legacy_v21=legacy_v21,
                create_camera_frame_iterator=_create_camera_frame_iterator,
                encode_videos_func=_encode_videos
            )

            # Return summary of all chunks
            total_frames = sum(r.get('result', {}).get('processing', {}).get('valid_frames', 0) for r in chunk_results)
            total_time = sum(r.get('result', {}).get('processing', {}).get('elapsed_time_s', 0) for r in chunk_results)
            total_parquet = sum(r.get('result', {}).get('files', {}).get('parquet_shards', 0) for r in chunk_results)
            total_video = sum(r.get('result', {}).get('files', {}).get('video_shards', 0) for r in chunk_results)

            # Update consolidated metadata after all chunks are processed
            logger.info("Updating consolidated metadata...")
            update_consolidated_metadata(
                out_dir, chunk_results, session_info, fps,
                len(chunk_results), total_frames, total_parquet, total_video,
                schema
            )

            return {
                'success': True,
                'mode': 'chunked',
                'num_chunks': len(chunk_results),
                'chunks': chunk_results,
                'processing': {
                    'total_frames': total_frames,
                    'valid_frames': total_frames,
                    'elapsed_time_s': total_time,
                },
                'files': {
                    'parquet_shards': total_parquet,
                    'video_shards': total_video,
                },
                'data_quality': {
                    'hand_valid_rate': 0.0,  # TODO: aggregate from chunks
                    'pose_valid_rate': 0.0,  # TODO: aggregate from chunks
                }
            }

        # Initialize Aria session (if not already done in split_csv mode)
        logger.info("Initializing Aria session...")
        session = AriaSession.from_paths(vrs_path, mps_dir)

        session_info = session.get_session_info()
        logger.info(f"Session capabilities: {session_info['capabilities']}")

        # Get camera frame iterator
        logger.info("Setting up camera frame iteration...")
        frame_iterator = _create_camera_frame_iterator(session, stride, max_frames)

        # Estimate total frames and FPS
        total_frames = _estimate_total_frames(session, stride, max_frames)
        if fps is None:
            fps = _estimate_fps(session)
            # Adjust FPS for stride (if stride=10, output FPS is 1/10 of camera FPS)
            fps = fps / stride

        logger.info(f"Estimated {total_frames} frames at {fps:.1f} FPS (stride={stride})")

        # Initialize progress tracking
        progress = ConversionProgress(total_frames)

        # Initialize data writers
        parquet_writer = ParquetWriter(
            output_dir=structure["data"],
            schema=schema,
            base_filename=f"episode_{episode_id:06d}"  # Standard LeRobot format
        )

        # Initialize haptic data loader if provided
        haptic_loader = None
        if haptic_csv is not None:
            logger.info(f"Loading haptic data from {haptic_csv} with offset {haptic_time_offset}s")
            haptic_loader = HapticDataLoader(haptic_csv, haptic_time_offset)

        # Get start timestamp from first frame
        # This will be the global dataset start timestamp (same for all episodes)
        start_timestamp_ns = 0
        try:
            first_frame = next(iter(_create_camera_frame_iterator(session, stride, 1)))
            start_timestamp_ns = first_frame[0]
            logger.info(f"Start timestamp: {start_timestamp_ns} ns")
            # Recreate iterator for actual processing
            frame_iterator = _create_camera_frame_iterator(session, stride, max_frames)
        except StopIteration:
            logger.warning("No frames found in stream")

        # For chunks (time_range_ns is set), use the chunk's start as the episode start
        # This ensures each episode's timestamps start from 0 (matching the video)
        # For full recordings (no time_range_ns), use the global start timestamp
        episode_start_timestamp_ns = start_timestamp_ns
        if time_range_ns is not None:
            episode_start_timestamp_ns = time_range_ns[0]
            logger.info(f"Using episode start timestamp: {episode_start_timestamp_ns} ns (relative to chunk start)")

        # Process frames
        logger.info("Starting frame processing...")
        if time_range_ns:
            logger.info(f"Time range filter: {time_range_ns[0]} to {time_range_ns[1] if time_range_ns[1] else 'end'} ns")
        processed_data = process_frames(
            session, frame_iterator, parquet_writer,
            progress, max_slop_ns,
            episode_start_timestamp_ns, haptic_loader,
            time_range_ns=time_range_ns,
            episode_id=episode_id,
            global_index_start=global_index_offset,
            fps=fps
        )
        
        # Finalize parquet writing
        logger.info("Finalizing parquet data...")
        parquet_metadata = parquet_writer.finalize()
        
        # Encode videos
        logger.info("Encoding videos...")
        # Use chunk_id for naming if provided, otherwise use episode_id
        video_id = chunk_id if chunk_id is not None else episode_id
        video_metadata = _encode_videos(
            processed_data["video_frames"],
            structure["video_main"],  # Use standard LeRobot video directory
            fps, video_id
        )
        
        # Write metadata files
        logger.info("Writing metadata files...")
        metadata_files = write_single_episode_metadata(
            structure, schema, session_info, fps,
            parquet_metadata, video_metadata, episode_id,
            processed_data["valid_frames"]
        )
        
        # Final validation
        logger.info("Validating output...")
        validation_results = validate_conversion_output(
            out_dir, processed_data["valid_frames"], False  # Always use standard format
        )
        
        if validation_results:
            logger.warning(f"Validation issues: {validation_results}")
        
        # Compilation summary
        elapsed_time = time.time() - start_time
        summary = {
            "success": True,
            "input": {
                "vrs_path": str(vrs_path),
                "mps_dir": str(mps_dir),
            },
            "output": {
                "directory": str(out_dir),
                "format": "v2.1" if legacy_v21 else "v3.0",
            },
            "processing": {
                "total_frames": total_frames,
                "valid_frames": processed_data["valid_frames"],
                "dropped_frames": processed_data["dropped_frames"],
                "fps": fps,
                "elapsed_time_s": elapsed_time,
            },
            "data_quality": {
                "hand_valid_rate": processed_data["hand_stats"]["valid_rate"],
                "pose_valid_rate": processed_data["pose_stats"]["valid_rate"],
                "nan_rates": processed_data["nan_rates"],
            },
            "files": {
                "parquet_shards": len(parquet_metadata["shards"]),
                "video_shards": len(video_metadata),
                "metadata_files": metadata_files,
            },
            "validation": validation_results,
        }
        
        logger.info("=== Conversion Completed Successfully ===")
        logger.info(f"Processed: {processed_data['valid_frames']}/{total_frames} frames")
        logger.info(f"Output: {len(parquet_metadata['shards'])} parquet + {len(video_metadata)} video files")
        logger.info(f"Time: {elapsed_time:.1f}s ({processed_data['valid_frames']/elapsed_time:.1f} fps)")
        
        # Generate visualization video if requested
        if create_visualization:
            logger.info("Generating visualization video...")
            try:
                viz_output_path = out_dir / "visualization_with_annotations.mp4"
                create_visualization_video(
                    vrs_path=str(vrs_path),
                    mps_dir=str(mps_dir),
                    output_path=str(viz_output_path),
                    max_frames=max_frames,
                    stride=stride,
                    rotate_image=rotate_visualization,
                    haptic_loader=haptic_loader,
                    start_timestamp_ns=start_timestamp_ns
                )
                summary["visualization_video"] = str(viz_output_path)
                logger.info(f"Visualization video created: {viz_output_path}")
            except Exception as e:
                logger.warning(f"Failed to create visualization video: {e}")
                summary["visualization_error"] = str(e)
        
        return summary
        
    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise


def _create_camera_frame_iterator(session: AriaSession, 
                                 stride: int = 1,
                                 max_frames: Optional[int] = None) -> Iterator[Tuple[int, np.ndarray]]:
    """Create iterator over camera-rgb frames"""
    
    if not hasattr(session, 'vrs_data_provider') or session.vrs_data_provider is None:
        raise RuntimeError("VRS data provider not available")
    
    # Get camera-rgb stream
    camera_label = "camera-rgb"
    stream_id = None
    
    try:
        stream_id = session.vrs_data_provider.get_stream_id_from_label(camera_label)
    except Exception:
        # Fallback: search through all streams
        all_streams = session.vrs_data_provider.get_all_streams()
        for sid in all_streams:
            try:
                label = session.vrs_data_provider.get_label_from_stream_id(sid)
                if camera_label in label or "rgb" in label.lower():
                    stream_id = sid
                    break
            except Exception:
                continue
    
    if stream_id is None:
        raise RuntimeError(f"Camera stream '{camera_label}' not found")
    
    logger.info(f"Using camera stream: {stream_id}")
    
    # Get total number of frames in stream
    try:
        # Get actual frame count from VRS stream
        stream_id_obj = session.vrs_data_provider.get_stream_id_from_label(camera_label)
        total_stream_frames = session.vrs_data_provider.get_num_data(stream_id_obj)
        logger.info(f"Got actual frame count from VRS: {total_stream_frames}")
    except Exception as e:
        logger.warning(f"Could not get actual frame count: {e}")
        # Fallback: estimate from trajectory but with better logic
        info = session.get_session_info()
        if 'trajectory_time_range_ns' in info:
            start_ns, end_ns = info['trajectory_time_range_ns']
            duration_s = (end_ns - start_ns) / 1e9
            # Try to estimate actual FPS from a small sample first
            try:
                sample_count = 0
                sample_timestamps = []
                for frame_idx in range(0, min(100, 10000), 10):  # Sample every 10th frame, up to 100 samples
                    try:
                        result = session.vrs_data_provider.get_image_data_by_index(stream_id, frame_idx)
                        if result is not None:
                            image_data, image_record = result[0], result[1]
                            sample_timestamps.append(image_record.capture_timestamp_ns)
                            sample_count += 1
                    except:
                        continue
                    if sample_count >= 10:  # Enough samples
                        break
                
                if len(sample_timestamps) >= 2:
                    sample_duration = (sample_timestamps[-1] - sample_timestamps[0]) / 1e9
                    estimated_fps = (len(sample_timestamps) - 1) / sample_duration if sample_duration > 0 else 10
                    total_stream_frames = int(duration_s * estimated_fps)
                    logger.info(f"Estimated FPS from samples: {estimated_fps:.2f}, frames: {total_stream_frames}")
                else:
                    total_stream_frames = int(duration_s * 10)  # Conservative 10 FPS fallback
                    logger.warning(f"Using conservative 10 FPS estimate: {total_stream_frames}")
            except Exception as sample_error:
                logger.warning(f"Sample-based estimation failed: {sample_error}")
                total_stream_frames = int(duration_s * 10)  # Conservative fallback
        else:
            total_stream_frames = 2000  # Conservative fallback
    
    logger.info(f"Estimated {total_stream_frames} frames in camera stream")
    
    # Iterate over frames
    frame_count = 0
    processed_count = 0
    
    for frame_idx in range(0, total_stream_frames, stride):
        if max_frames and processed_count >= max_frames:
            break
        
        try:
            # Get image data by index
            result = session.vrs_data_provider.get_image_data_by_index(stream_id, frame_idx)
            if result is None:
                continue
            
            # Unpack result
            image_data, image_record = result[0], result[1]
            timestamp_ns = image_record.capture_timestamp_ns
            
            # Convert to numpy array
            image_array = image_data.to_numpy_array()
            
            # Ensure RGB format
            image_array = ensure_rgb_format(image_array, "rgb")  # VRS provides RGB format
            
            yield timestamp_ns, image_array
            processed_count += 1
            
        except Exception as e:
            logger.debug(f"Failed to get frame {frame_idx}: {e}")
            continue
        
        frame_count += 1
    
    logger.info(f"Frame iterator completed: {processed_count} valid frames from {frame_count} attempts")


def _estimate_total_frames(session: AriaSession, stride: int, max_frames: Optional[int]) -> int:
    """Estimate total number of frames to process"""
    try:
        # Get actual frame count from VRS stream using correct API
        camera_label = "camera-rgb"
        stream_id_obj = session.vrs_data_provider.get_stream_id_from_label(camera_label)
        actual_frames = session.vrs_data_provider.get_num_data(stream_id_obj)
        total_frames = (actual_frames + stride - 1) // stride  # Account for stride
        logger.info(f"Got actual frame count for estimation: {actual_frames}, with stride {stride} = {total_frames}")
    except Exception as e:
        logger.warning(f"Could not get actual frame count for estimation: {e}")
        # Fallback: estimate from session info with better logic
        info = session.get_session_info()
        if 'trajectory_time_range_ns' in info:
            start_ns, end_ns = info['trajectory_time_range_ns']
            duration_s = (end_ns - start_ns) / 1e9
            # Use conservative 10 FPS estimate instead of 30
            total_frames = int(duration_s * 10 / stride)
            logger.warning(f"Using fallback estimation: {duration_s:.1f}s * 10fps / {stride} = {total_frames}")
        else:
            total_frames = max(1000 // stride, 1)  # Conservative fallback
    
    if max_frames:
        total_frames = min(total_frames, max_frames)
    
    return total_frames


def _estimate_fps(session: AriaSession) -> float:
    """Estimate camera FPS from session data"""
    try:
        # Get a sample of timestamps using consecutive frames (stride=1)
        sample_timestamps = []
        frame_iterator = _create_camera_frame_iterator(session, stride=1, max_frames=100)
        
        for timestamp_ns, _ in frame_iterator:
            sample_timestamps.append(timestamp_ns)
            if len(sample_timestamps) >= 50:  # Enough for good estimate
                break
        
        if len(sample_timestamps) >= 2:
            fps = estimate_fps_from_timestamps(sample_timestamps)
            logger.info(f"Estimated FPS from camera frames: {fps:.1f}")
            return fps
    
    except Exception as e:
        logger.warning(f"FPS estimation failed: {e}")
    
    # Fallback
    return 30.0


def _encode_videos(video_frames: List[Tuple[int, np.ndarray]],
                  videos_dir: Path,
                  fps: float,
                  episode_id: int) -> List[Dict[str, Any]]:
    """Encode video frames to MP4 files"""

    if not video_frames:
        logger.warning("No video frames to encode")
        return []

    logger.info(f"Encoding {len(video_frames)} video frames...")

    # Get video shape from first frame (height, width, channels)
    _, first_frame = video_frames[0]
    video_shape = first_frame.shape  # (H, W, C)

    # Standard LeRobot format: episode-based video files
    from .video import encode_image_sequence

    video_path = videos_dir / f"episode_{episode_id:06d}.mp4"
    metadata = encode_image_sequence(
        iter(video_frames), video_path, fps,
        codec="h264", quality="medium"
    )

    # Add video shape to metadata
    metadata["video_shape"] = video_shape

    return [metadata]


