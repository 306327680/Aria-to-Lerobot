"""
LeRobot Dataset Schema Definitions

Defines the data schema for LeRobot v3.0 and v2.1 formats with strict
column names, shapes, and dtypes for consistent data access.
"""

from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


@dataclass
class ColumnSpec:
    """Column specification with shape and dtype"""
    shape: Tuple[int, ...]
    dtype: str
    description: str = ""
    
    def validate_data(self, data: np.ndarray) -> bool:
        """Validate data against spec"""
        if data.dtype != np.dtype(self.dtype):
            return False
        if data.shape[-len(self.shape):] != self.shape:
            return False
        return True


@dataclass
class LeRobotSchema:
    """LeRobot dataset schema definition"""
    version: str
    columns: Dict[str, ColumnSpec]
    required_columns: List[str]
    video_columns: List[str]
    
    def get_parquet_schema(self) -> Dict[str, Any]:
        """Get PyArrow schema for parquet files"""
        import pyarrow as pa
        
        schema_fields = []
        for col_name, col_spec in self.columns.items():
            if col_name in self.video_columns:
                continue  # Videos stored separately
                
            # Convert numpy dtype to pyarrow type
            if col_spec.dtype == "int64":
                pa_type = pa.int64()
            elif col_spec.dtype == "int32":
                pa_type = pa.int32()
            elif col_spec.dtype == "float64":
                pa_type = pa.float64()
            elif col_spec.dtype == "float32":
                pa_type = pa.float32()
            elif col_spec.dtype == "bool":
                pa_type = pa.bool_()
            else:
                pa_type = pa.string()
            
            # Handle array columns
            if len(col_spec.shape) > 0:
                # Flatten arrays for parquet storage
                total_elements = np.prod(col_spec.shape)
                if total_elements > 1:
                    pa_type = pa.list_(pa_type, total_elements)
            
            schema_fields.append(pa.field(col_name, pa_type))
        
        return pa.schema(schema_fields)
    
    def get_info_json(self, fps: float, episode_count: int, total_frames: int = 0,
                      video_shape: Tuple[int, int, int] = (224, 224, 3), **kwargs) -> Dict[str, Any]:
        """Generate meta/info.json content compatible with LeRobotDataset

        Args:
            fps: Frames per second
            episode_count: Number of episodes
            total_frames: Total frame count across all episodes
            video_shape: Video frame shape (height, width, channels)
            **kwargs: Additional metadata fields
        """

        # Generate v2.1-compatible features schema (without _type for visualize_dataset compatibility)
        features = {}
        for col_name, col_spec in self.columns.items():
            if len(col_spec.shape) == 0:
                # Scalar values - use shape: [1] for v2.1 compatibility
                features[col_name] = {
                    "dtype": col_spec.dtype,
                    "shape": [1],
                    "names": None
                }
            else:
                # Array values
                shape_list = list(col_spec.shape)
                features[col_name] = {
                    "dtype": col_spec.dtype,
                    "shape": shape_list,
                    "names": None
                }

        # Add video features with full v2.1 video metadata
        h, w, c = video_shape
        for video_col in self.video_columns:
            features[video_col] = {
                "dtype": "video",
                "shape": [h, w, c],  # H, W, C order for v2.1
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": h,
                    "video.width": w,
                    "video.codec": "libx264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": fps,
                    "video.channels": c,
                    "has_audio": False
                }
            }

        info = {
            # Core metadata - v3.0 format
            "codebase_version": "v3.0",
            "total_episodes": episode_count,
            "total_frames": total_frames,
            "total_tasks": 1,  # Single task (aria_recording)
            "total_videos": len(self.video_columns),
            "total_chunks": 1,  # Single chunk for now
            "chunks_size": 1000,
            "fps": fps,

            # Data paths
            "data_path": "data/chunk-{chunk_index:03d}/episode_{file_index:06d}.parquet",
            "video_path": "videos/chunk-{chunk_index:03d}/{video_key}/episode_{file_index:06d}.mp4",

            # v2.1 features schema (compatible with visualize_dataset)
            "features": features
        }

        # Add optional robot_type if provided
        if "robot_type" in kwargs:
            info["robot_type"] = kwargs["robot_type"]
        else:
            info["robot_type"] = "aria"  # Default for Aria recordings

        # Add train split if provided
        if "splits" in kwargs:
            info["splits"] = kwargs["splits"]
        else:
            info["splits"] = {"train": f"0:{episode_count}"}

        # Update with additional kwargs
        for key, value in kwargs.items():
            if key not in info and key != "video_shape":
                info[key] = value

        return info


# LeRobot v3.0 Schema Definition
LEROBOT_V3_SCHEMA = LeRobotSchema(
    version="3.0",
    columns={
        # Core temporal columns
        "index": ColumnSpec((), "int64", "Global index across all episodes"),
        "episode_index": ColumnSpec((), "int64", "Episode index (same as episode_id)"),
        "episode_chunk": ColumnSpec((), "int64", "Episode chunk index for sharding"),
        "frame_index": ColumnSpec((), "int64", "Frame index within episode (same as step)"),
        "task_index": ColumnSpec((), "int64", "Task index"),
        "timestamp": ColumnSpec((), "float64", "Timestamp in seconds"),
        "timestamp_ns": ColumnSpec((), "int64", "Timestamp in nanoseconds (device time)"),
        "episode_id": ColumnSpec((), "int32", "Episode identifier"),
        "step": ColumnSpec((), "int64", "Step within episode (0-indexed)"),

        # Standard LeRobot observation.state: hand landmarks + pose (63 + 16 = 79 dims)
        "observation.state": ColumnSpec((79,), "float32", "Hand landmarks (63) + device pose (16)"),

        # Standard LeRobot action: tactile forces (left_total + right_total + left_xyz + right_xyz = 1+1+3+3 = 8 dims)
        "action": ColumnSpec((8,), "float32", "Tactile forces: [left_total, right_total, left_xyz(3), right_xyz(3)]"),

        # Keep original detailed columns for reference
        "status.hands.landmarks_world": ColumnSpec((63,), "float32", "Hand landmarks in world coords (21×3 flattened)"),
        "status.hands.valid_left": ColumnSpec((), "bool", "Left hand validity flag"),
        "status.hands.valid_right": ColumnSpec((), "bool", "Right hand validity flag"),

        "action.tactile.left.force_total": ColumnSpec((), "float32", "Left hand total force"),
        "action.tactile.right.force_total": ColumnSpec((), "float32", "Right hand total force"),
        "action.tactile.left.force_xyz": ColumnSpec((3,), "float32", "Left hand force vector"),
        "action.tactile.right.force_xyz": ColumnSpec((3,), "float32", "Right hand force vector"),

        # Tactile action masks
        "action.mask.tactile.left.force_total": ColumnSpec((), "bool", "Left force total mask"),
        "action.mask.tactile.right.force_total": ColumnSpec((), "bool", "Right force total mask"),
        "action.mask.tactile.left.force_xyz": ColumnSpec((3,), "bool", "Left force vector mask"),
        "action.mask.tactile.right.force_xyz": ColumnSpec((3,), "bool", "Right force vector mask"),

        # Device pose
        "status.pose.world_device": ColumnSpec((16,), "float32", "World to device transform (4×4 flattened)"),
        "status.pose.valid": ColumnSpec((), "bool", "Pose validity flag"),
    },
    required_columns=[
        "index", "episode_index", "episode_chunk", "frame_index", "task_index",
        "timestamp", "timestamp_ns", "episode_id", "step",
        "observation.state", "action"  # Standard LeRobot columns
    ],
    video_columns=["observation.images.main"]
)

# LeRobot v2.1 Schema (compatible subset)
LEROBOT_V21_SCHEMA = LeRobotSchema(
    version="2.1",
    columns={
        # Core temporal columns (same as v3)
        "index": ColumnSpec((), "int64", "Global index across all episodes"),
        "episode_index": ColumnSpec((), "int64", "Episode index (same as episode_id)"),
        "frame_index": ColumnSpec((), "int64", "Frame index within episode (same as step)"),
        "task_index": ColumnSpec((), "int64", "Task index"),
        "timestamp": ColumnSpec((), "float64", "Timestamp in seconds"),
        "timestamp_ns": ColumnSpec((), "int64", "Timestamp in nanoseconds (device time)"),
        "episode_id": ColumnSpec((), "int32", "Episode identifier"),
        "step": ColumnSpec((), "int64", "Step within episode (0-indexed)"),

        # Standard LeRobot observation.state: hand landmarks + pose (63 + 16 = 79 dims)
        "observation.state": ColumnSpec((79,), "float32", "Hand landmarks (63) + device pose (16)"),

        # Standard LeRobot action: tactile forces (left_total + right_total + left_xyz + right_xyz = 1+1+3+3 = 8 dims)
        "action": ColumnSpec((8,), "float32", "Tactile forces: [left_total, right_total, left_xyz(3), right_xyz(3)]"),

        # Keep original detailed columns for reference
        "status.hands.landmarks_world": ColumnSpec((63,), "float32", "Hand landmarks in world coords (21×3 flattened)"),
        "status.hands.valid_left": ColumnSpec((), "bool", "Left hand validity flag"),
        "status.hands.valid_right": ColumnSpec((), "bool", "Right hand validity flag"),

        "action.tactile.left.force_total": ColumnSpec((), "float32", "Left hand total force"),
        "action.tactile.right.force_total": ColumnSpec((), "float32", "Right hand total force"),
        "action.tactile.left.force_xyz": ColumnSpec((3,), "float32", "Left hand force vector"),
        "action.tactile.right.force_xyz": ColumnSpec((3,), "float32", "Right hand force vector"),

        # Tactile action masks
        "action.mask.tactile.left.force_total": ColumnSpec((), "bool", "Left force total mask"),
        "action.mask.tactile.right.force_total": ColumnSpec((), "bool", "Right force total mask"),
        "action.mask.tactile.left.force_xyz": ColumnSpec((3,), "bool", "Left force vector mask"),
        "action.mask.tactile.right.force_xyz": ColumnSpec((3,), "bool", "Right force vector mask"),

        # Device pose
        "status.pose.world_device": ColumnSpec((16,), "float32", "World to device transform (4×4 flattened)"),
        "status.pose.valid": ColumnSpec((), "bool", "Pose validity flag"),
    },
    required_columns=[
        "index", "episode_index", "frame_index", "task_index",
        "timestamp", "timestamp_ns", "episode_id", "step",
        "observation.state", "action"  # Standard LeRobot columns
    ],
    video_columns=["observation.images.main"]
)


def flatten_hand_landmarks(left_landmarks: Optional[np.ndarray], 
                          right_landmarks: Optional[np.ndarray]) -> Tuple[np.ndarray, bool, bool]:
    """
    Flatten hand landmarks for storage
    
    Args:
        left_landmarks: Left hand landmarks (21, 3) or None
        right_landmarks: Right hand landmarks (21, 3) or None
    
    Returns:
        (flattened_landmarks, valid_left, valid_right)
        flattened_landmarks: (63,) array with left hand landmarks (21×3 flattened)
    """
    flattened = np.full(63, np.nan, dtype=np.float32)
    valid_left = False
    valid_right = False
    
    if left_landmarks is not None and left_landmarks.shape == (21, 3):
        flattened[:63] = left_landmarks.flatten()
        valid_left = not np.any(np.isnan(left_landmarks))
    
    # Note: Only storing left hand landmarks in the 63-element array
    # Right hand would need separate column or extended array
    if right_landmarks is not None and right_landmarks.shape == (21, 3):
        valid_right = not np.any(np.isnan(right_landmarks))
        # For now, right hand landmarks are tracked via valid flag only
        # Could extend to 126-element array if needed
    
    return flattened, valid_left, valid_right


def unflatten_hand_landmarks(flattened: np.ndarray) -> np.ndarray:
    """
    Unflatten hand landmarks from storage
    
    Args:
        flattened: (63,) flattened left hand landmarks
    
    Returns:
        landmarks: (21, 3) left hand landmarks
    """
    if len(flattened) != 63:
        raise ValueError(f"Expected 63 elements, got {len(flattened)}")
    
    return flattened.reshape(21, 3)


def create_tactile_placeholders(force_left: Optional[Tuple[float, np.ndarray]] = None,
                               force_right: Optional[Tuple[float, np.ndarray]] = None) -> Dict[str, Any]:
    """
    Create tactile action data (real values or NaN placeholders)

    Args:
        force_left: Tuple of (magnitude, xyz_array) for left sensor (optional)
        force_right: Tuple of (magnitude, xyz_array) for right sensor (optional)

    Returns:
        Dictionary with tactile data columns and masks
    """
    if force_left is not None and force_right is not None:
        # Real haptic data available - unpack tuples
        left_mag, left_xyz = force_left
        right_mag, right_xyz = force_right

        return {
            "action.tactile.left.force_total": np.float32(left_mag),
            "action.tactile.right.force_total": np.float32(right_mag),
            "action.tactile.left.force_xyz": left_xyz.astype(np.float32),
            "action.tactile.right.force_xyz": right_xyz.astype(np.float32),
            "action.mask.tactile.left.force_total": True,
            "action.mask.tactile.right.force_total": True,
            "action.mask.tactile.left.force_xyz": np.array([True, True, True]),
            "action.mask.tactile.right.force_xyz": np.array([True, True, True]),
        }
    else:
        # Fallback to NaN placeholders
        return {
            "action.tactile.left.force_total": np.float32(np.nan),
            "action.tactile.right.force_total": np.float32(np.nan),
            "action.tactile.left.force_xyz": np.full(3, np.nan, dtype=np.float32),
            "action.tactile.right.force_xyz": np.full(3, np.nan, dtype=np.float32),
            "action.mask.tactile.left.force_total": False,
            "action.mask.tactile.right.force_total": False,
            "action.mask.tactile.left.force_xyz": np.array([False, False, False]),
            "action.mask.tactile.right.force_xyz": np.array([False, False, False]),
        }


def flatten_pose_matrix(pose_matrix: np.ndarray) -> np.ndarray:
    """Flatten 4×4 pose matrix for storage"""
    if pose_matrix.shape != (4, 4):
        raise ValueError(f"Expected (4, 4) matrix, got {pose_matrix.shape}")
    return pose_matrix.flatten().astype(np.float32)


def unflatten_pose_matrix(flattened: np.ndarray) -> np.ndarray:
    """Unflatten pose matrix from storage"""
    if len(flattened) != 16:
        raise ValueError(f"Expected 16 elements, got {len(flattened)}")
    return flattened.reshape(4, 4)


def validate_schema_data(data: Dict[str, Any], schema: LeRobotSchema) -> List[str]:
    """
    Validate data against schema
    
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Check required columns
    for col in schema.required_columns:
        if col not in data:
            errors.append(f"Missing required column: {col}")
    
    # Check column specs
    for col_name, value in data.items():
        if col_name not in schema.columns:
            errors.append(f"Unknown column: {col_name}")
            continue
        
        spec = schema.columns[col_name]
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        
        if not spec.validate_data(value):
            errors.append(f"Column {col_name} validation failed: "
                         f"expected shape {spec.shape}, dtype {spec.dtype}, "
                         f"got shape {value.shape}, dtype {value.dtype}")
    
    return errors
