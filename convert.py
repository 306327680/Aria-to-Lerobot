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
from projectaria_tools.core.sensor_data import TimeDomain

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


logger = logging.getLogger(__name__)


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
           rotate_visualization: bool = False) -> Dict[str, Any]:
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
    
    validation_errors = validate_paths(vrs_path, mps_dir, out_dir, overwrite)
    if validation_errors:
        for error in validation_errors:
            logger.error(error)
        raise ValueError(f"Validation failed: {validation_errors}")
    
    # Create output structure
    structure = create_output_structure(out_dir, legacy_v21)
    schema = LEROBOT_V21_SCHEMA if legacy_v21 else LEROBOT_V3_SCHEMA
    
    start_time = time.time()
    episode_id = 0  # Default episode ID
    
    try:
        # Initialize Aria session
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
        
        logger.info(f"Estimated {total_frames} frames at {fps:.1f} FPS")
        
        # Initialize progress tracking
        progress = ConversionProgress(total_frames)
        
        # Initialize data writers
        parquet_writer = ParquetWriter(
            output_dir=structure["data"],
            schema=schema,
            base_filename=f"episode_{episode_id:06d}"  # Standard LeRobot format
        )
        
        # Process frames
        logger.info("Starting frame processing...")
        processed_data = _process_frames(
            session, frame_iterator, parquet_writer, 
            progress, max_slop_ns, schema
        )
        
        # Finalize parquet writing
        logger.info("Finalizing parquet data...")
        parquet_metadata = parquet_writer.finalize()
        
        # Encode videos
        logger.info("Encoding videos...")
        video_metadata = _encode_videos(
            processed_data["video_frames"], 
            structure["video_main"],  # Use standard LeRobot video directory
            fps, episode_id
        )
        
        # Write metadata files
        logger.info("Writing metadata files...")
        metadata_files = _write_metadata_files(
            structure, schema, session_info, fps, 
            parquet_metadata, video_metadata, episode_id
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
                    rotate_image=rotate_visualization
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
        from aria_api.timestamps import TimeDomain
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


def _process_frames(session: AriaSession,
                   frame_iterator: Iterator[Tuple[int, np.ndarray]],
                   parquet_writer: ParquetWriter,
                   progress: ConversionProgress,
                   max_slop_ns: int,
                   schema: LeRobotSchema) -> Dict[str, Any]:
    """Process camera frames and extract aligned data"""
    
    processed_data = {
        "valid_frames": 0,
        "dropped_frames": 0,
        "video_frames": [],
        "hand_stats": {"valid": 0, "total": 0},
        "pose_stats": {"valid": 0, "total": 0},
        "nan_rates": {},
    }
    
    step = 0
    episode_id = 0
    
    for timestamp_ns, image in frame_iterator:
        try:
            # Extract aligned data
            row_data = _extract_frame_data(
                session, timestamp_ns, step, episode_id, max_slop_ns
            )
            
            if row_data is None:
                processed_data["dropped_frames"] += 1
                progress.update(valid=False)
                continue
            
            # Add to parquet
            parquet_writer.add_row(row_data)
            
            # Store video frame
            processed_data["video_frames"].append((timestamp_ns, image))
            
            # Update statistics
            _update_processing_stats(row_data, processed_data)
            
            processed_data["valid_frames"] += 1
            step += 1
            progress.update(valid=True)
            
        except Exception as e:
            logger.debug(f"Failed to process frame at {timestamp_ns}: {e}")
            processed_data["dropped_frames"] += 1
            progress.update(valid=False)
            continue
    
    # Finalize statistics
    _finalize_processing_stats(processed_data)
    
    return processed_data


def _extract_frame_data(session: AriaSession,
                       timestamp_ns: int,
                       step: int,
                       episode_id: int,
                       max_slop_ns: int) -> Optional[Dict[str, Any]]:
    """Extract all aligned data for a single frame"""
    
    # Base temporal data
    row_data = {
        "timestamp": timestamp_ns / 1e9,  # Convert to seconds
        "timestamp_ns": timestamp_ns,
        "episode_id": np.int32(episode_id),
        "step": np.int64(step),
    }
    
    # Extract pose data
    pose = session.nearest_pose(timestamp_ns, max_slop_ns)
    if pose is not None:
        pose_matrix = pose.T_world_device.to_matrix()
        row_data["status.pose.world_device"] = flatten_pose_matrix(pose_matrix)
        row_data["status.pose.valid"] = True
    else:
        row_data["status.pose.world_device"] = np.full(16, np.nan, dtype=np.float32)
        row_data["status.pose.valid"] = False
    
    # Extract hand data using the hands module interface
    hands_data = session.nearest_hands(timestamp_ns, max_slop_ns)
    if hands_data is not None and pose is not None:
        # Get world coordinates using the hand projector
        world_landmarks = session.hand_projector.landmarks_in_world(
            timestamp_ns, pose.T_world_device, max_slop_ns=max_slop_ns
        )
        
        left_landmarks = world_landmarks.get("left")
        right_landmarks = world_landmarks.get("right")
        
        # For now, store only left hand landmarks (as per schema design)
        flattened, valid_left, valid_right = flatten_hand_landmarks(left_landmarks, right_landmarks)
        row_data["status.hands.landmarks_world"] = flattened
        row_data["status.hands.valid_left"] = valid_left
        row_data["status.hands.valid_right"] = valid_right
    else:
        # No hand data available
        row_data["status.hands.landmarks_world"] = np.full(63, np.nan, dtype=np.float32)
        row_data["status.hands.valid_left"] = False
        row_data["status.hands.valid_right"] = False
    
    # Add tactile placeholders
    tactile_data = create_tactile_placeholders()
    row_data.update(tactile_data)
    
    return row_data


def _update_processing_stats(row_data: Dict[str, Any], processed_data: Dict[str, Any]):
    """Update processing statistics"""
    
    # Hand statistics
    processed_data["hand_stats"]["total"] += 1
    if row_data["status.hands.valid_left"] or row_data["status.hands.valid_right"]:
        processed_data["hand_stats"]["valid"] += 1
    
    # Pose statistics  
    processed_data["pose_stats"]["total"] += 1
    if row_data["status.pose.valid"]:
        processed_data["pose_stats"]["valid"] += 1


def _finalize_processing_stats(processed_data: Dict[str, Any]):
    """Finalize processing statistics"""
    
    # Compute rates
    if processed_data["hand_stats"]["total"] > 0:
        processed_data["hand_stats"]["valid_rate"] = (
            processed_data["hand_stats"]["valid"] / processed_data["hand_stats"]["total"]
        )
    else:
        processed_data["hand_stats"]["valid_rate"] = 0.0
    
    if processed_data["pose_stats"]["total"] > 0:
        processed_data["pose_stats"]["valid_rate"] = (
            processed_data["pose_stats"]["valid"] / processed_data["pose_stats"]["total"]
        )
    else:
        processed_data["pose_stats"]["valid_rate"] = 0.0


def _encode_videos(video_frames: List[Tuple[int, np.ndarray]],
                  videos_dir: Path,
                  fps: float,
                  episode_id: int) -> List[Dict[str, Any]]:
    """Encode video frames to MP4 files"""
    
    if not video_frames:
        logger.warning("No video frames to encode")
        return []
    
    logger.info(f"Encoding {len(video_frames)} video frames...")
    
    # Standard LeRobot format: episode-based video files
    from .video import encode_image_sequence
    
    video_path = videos_dir / f"episode_{episode_id:06d}.mp4"
    metadata = encode_image_sequence(
        iter(video_frames), video_path, fps,
        codec="h264", quality="medium"
    )
    return [metadata]


def _write_metadata_files(structure: Dict[str, Path],
                         schema: LeRobotSchema,
                         session_info: Dict[str, Any],
                         fps: float,
                         parquet_metadata: Dict[str, Any],
                         video_metadata: List[Dict[str, Any]],
                         episode_id: int) -> List[str]:
    """Write metadata files for the dataset"""
    
    written_files = []
    
    # Standard LeRobot metadata structure
    
    # meta/info.json
    info_data = schema.get_info_json(
        fps=fps,
        episode_count=1,
        total_frames=parquet_metadata["total_rows"],
        aria_session_info=session_info,
        conversion_time=time.time(),
        video_shards=len(video_metadata),
        parquet_shards=len(parquet_metadata["shards"]),
    )
    
    info_file = structure["meta"] / "info.json"
    with open(info_file, 'w') as f:
        json.dump(info_data, f, indent=2)
    written_files.append(str(info_file))
    
    # meta/stats.json
    stats_file = structure["meta"] / "stats.json"
    with open(stats_file, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(v) for v in obj]
            return obj
        
        json.dump(convert_numpy_types(parquet_metadata["statistics"]), f, indent=2)
    written_files.append(str(stats_file))
    
    # meta/episodes.jsonl (standard LeRobot format)
    episodes_file = structure["meta"] / "episodes.jsonl"
    with open(episodes_file, 'w') as f:
        episode_info = {
            "episode_index": episode_id,
            "episode_id": f"episode_{episode_id:06d}",
            "length": parquet_metadata["total_rows"],
            "timestamp_start": parquet_metadata["statistics"].get("timestamp", {}).get("min", 0),
            "timestamp_end": parquet_metadata["statistics"].get("timestamp", {}).get("max", 0),
            "task": "aria_recording"  # Default task name
        }
        f.write(json.dumps(episode_info) + "\n")
    written_files.append(str(episodes_file))
    
    # meta/tasks.jsonl (standard LeRobot format)
    tasks_file = structure["meta"] / "tasks.jsonl"
    with open(tasks_file, 'w') as f:
        task_info = {
            "task_index": 0,
            "task_name": "aria_recording",
            "task_description": "Project Aria multi-modal recording converted to LeRobot format"
        }
        f.write(json.dumps(task_info) + "\n")
    written_files.append(str(tasks_file))
    
    logger.info(f"Written metadata files: {written_files}")
    return written_files
