"""
Video encoding utilities for aria2lerobot

Handles H.264/H.265 video encoding for LeRobot dataset video columns.
Supports both streaming and batch encoding modes.
"""

import logging
from pathlib import Path
from typing import Iterator, Tuple, Optional, Dict, Any, List
import numpy as np
import cv2
from contextlib import contextmanager


logger = logging.getLogger(__name__)


class VideoEncoder:
    """Video encoder for LeRobot datasets"""
    
    def __init__(self, 
                 output_path: Path,
                 fps: float = 30.0,
                 codec: str = "h264",
                 quality: str = "medium",
                 color_format: str = "rgb"):
        """
        Initialize video encoder
        
        Args:
            output_path: Output video file path
            fps: Frames per second
            codec: Video codec ("h264", "h265")
            quality: Encoding quality ("low", "medium", "high")
            color_format: Input color format ("rgb", "bgr")
        """
        self.output_path = Path(output_path)
        self.fps = fps
        self.codec = codec
        self.quality = quality
        self.color_format = color_format.lower()
        
        self.writer: Optional[cv2.VideoWriter] = None
        self.frame_count = 0
        self.is_initialized = False
        
        # Codec mapping
        self.codec_map = {
            "h264": cv2.VideoWriter_fourcc(*'H264'),
            "h265": cv2.VideoWriter_fourcc(*'HEVC'),
            "mp4v": cv2.VideoWriter_fourcc(*'mp4v'),
        }
        
        # Quality settings (CRF-like values for OpenCV)
        self.quality_map = {
            "low": {"bitrate": 500000},      # ~500kbps
            "medium": {"bitrate": 2000000},  # ~2Mbps  
            "high": {"bitrate": 5000000},    # ~5Mbps
        }
    
    def _get_fourcc(self) -> int:
        """Get OpenCV fourcc code"""
        if self.codec in self.codec_map:
            return self.codec_map[self.codec]
        else:
            logger.warning(f"Unknown codec {self.codec}, using H264")
            return self.codec_map["h264"]
    
    def initialize(self, frame_shape: Tuple[int, int, int]):
        """Initialize video writer with first frame"""
        height, width, channels = frame_shape
        
        if channels != 3:
            raise ValueError(f"Expected 3-channel image, got {channels}")
        
        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize VideoWriter
        fourcc = self._get_fourcc()
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (width, height),
            isColor=True
        )
        
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to initialize video writer for {self.output_path}")
        
        self.is_initialized = True
        logger.info(f"Initialized video encoder: {width}×{height} @ {self.fps} fps -> {self.output_path}")
    
    def add_frame(self, frame: np.ndarray):
        """Add a frame to the video"""
        if frame.dtype != np.uint8:
            # Convert to uint8 if needed
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = np.clip(frame, 0, 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        # Initialize on first frame
        if not self.is_initialized:
            self.initialize(frame.shape)
        
        # Convert color format if needed
        if self.color_format == "rgb":
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            # Already BGR
            frame_bgr = frame
        
        # Write frame
        if self.writer is not None:
            self.writer.write(frame_bgr)
            self.frame_count += 1
        else:
            raise RuntimeError("Video writer not initialized")
    
    def finalize(self) -> Dict[str, Any]:
        """Finalize video encoding and return metadata"""
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        
        # Get file size
        file_size = self.output_path.stat().st_size if self.output_path.exists() else 0
        
        metadata = {
            "path": str(self.output_path),
            "frame_count": self.frame_count,
            "fps": self.fps,
            "codec": self.codec,
            "quality": self.quality,
            "file_size_bytes": file_size,
            "duration_s": self.frame_count / self.fps if self.fps > 0 else 0,
        }
        
        logger.info(f"Video encoding completed: {self.frame_count} frames, {file_size} bytes")
        return metadata


@contextmanager
def video_encoder_context(*args, **kwargs):
    """Context manager for video encoder"""
    encoder = VideoEncoder(*args, **kwargs)
    try:
        yield encoder
    finally:
        encoder.finalize()


def encode_image_sequence(images: Iterator[Tuple[int, np.ndarray]], 
                         output_path: Path,
                         fps: float,
                         **encoder_kwargs) -> Dict[str, Any]:
    """
    Encode a sequence of images to video
    
    Args:
        images: Iterator of (timestamp_ns, image) tuples
        output_path: Output video path
        fps: Target FPS
        **encoder_kwargs: Additional encoder parameters
    
    Returns:
        Encoding metadata
    """
    with video_encoder_context(output_path, fps=fps, **encoder_kwargs) as encoder:
        frame_count = 0
        timestamps = []
        
        for timestamp_ns, image in images:
            encoder.add_frame(image)
            timestamps.append(timestamp_ns)
            frame_count += 1
        
        metadata = encoder.finalize()
        metadata.update({
            "timestamps_ns": timestamps,
            "actual_frame_count": frame_count,
        })
        
        return metadata


def create_video_chunks(images: Iterator[Tuple[int, np.ndarray]], 
                       chunk_size: int = 1000) -> Iterator[Tuple[int, list]]:
    """
    Split image sequence into chunks for processing
    
    Args:
        images: Iterator of (timestamp_ns, image) tuples
        chunk_size: Number of frames per chunk
    
    Yields:
        (chunk_id, chunk_frames) tuples
    """
    chunk_id = 0
    chunk_frames = []
    
    for timestamp_ns, image in images:
        chunk_frames.append((timestamp_ns, image))
        
        if len(chunk_frames) >= chunk_size:
            yield chunk_id, chunk_frames
            chunk_id += 1
            chunk_frames = []
    
    # Yield remaining frames
    if chunk_frames:
        yield chunk_id, chunk_frames


def encode_video_shards(images: Iterator[Tuple[int, np.ndarray]],
                       output_dir: Path,
                       base_name: str = "main",
                       fps: float = 30.0,
                       shard_size: int = 10000,
                       **encoder_kwargs) -> List[Dict[str, Any]]:
    """
    Encode image sequence to multiple video shards
    
    Args:
        images: Iterator of (timestamp_ns, image) tuples
        output_dir: Output directory for video shards
        base_name: Base name for video files (e.g., "main" -> "main_000.mp4")
        fps: Target FPS
        shard_size: Frames per shard
        **encoder_kwargs: Additional encoder parameters
    
    Returns:
        List of shard metadata dictionaries
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    shard_metadata = []
    
    for chunk_id, chunk_frames in create_video_chunks(images, shard_size):
        # Generate shard filename
        shard_filename = f"{base_name}_{chunk_id:03d}.mp4"
        shard_path = output_dir / shard_filename
        
        logger.info(f"Encoding shard {chunk_id}: {len(chunk_frames)} frames -> {shard_filename}")
        
        # Encode chunk
        chunk_iter = iter(chunk_frames)
        metadata = encode_image_sequence(chunk_iter, shard_path, fps, **encoder_kwargs)
        metadata["shard_id"] = chunk_id
        metadata["shard_filename"] = shard_filename
        
        shard_metadata.append(metadata)
    
    logger.info(f"Encoded {len(shard_metadata)} video shards")
    return shard_metadata


def validate_video_file(video_path: Path, expected_frames: Optional[int] = None) -> Dict[str, Any]:
    """
    Validate video file and extract metadata
    
    Args:
        video_path: Path to video file
        expected_frames: Expected number of frames (optional)
    
    Returns:
        Validation results and metadata
    """
    if not video_path.exists():
        return {"valid": False, "error": "File does not exist"}
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            return {"valid": False, "error": "Cannot open video file"}
        
        # Extract metadata
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        # Check frame count if expected
        frame_count_ok = True
        if expected_frames is not None:
            frame_count_ok = abs(frame_count - expected_frames) <= max(1, expected_frames * 0.01)  # 1% tolerance
        
        file_size = video_path.stat().st_size
        
        result = {
            "valid": frame_count_ok,
            "frame_count": frame_count,
            "expected_frames": expected_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "file_size_bytes": file_size,
            "duration_s": frame_count / fps if fps > 0 else 0,
        }
        
        if not frame_count_ok:
            result["error"] = f"Frame count mismatch: expected {expected_frames}, got {frame_count}"
        
        return result
        
    except Exception as e:
        return {"valid": False, "error": str(e)}


def ensure_rgb_format(image: np.ndarray, source_format: str = "auto") -> np.ndarray:
    """
    Ensure image is in RGB format
    
    Args:
        image: Input image
        source_format: Source format ("rgb", "bgr", "auto")
    
    Returns:
        RGB image
    """
    if source_format == "auto":
        # Heuristic: if aria_api returns BGR (common with OpenCV), convert to RGB
        # This is dataset/implementation dependent
        source_format = "bgr"  # Assume BGR from aria_api
    
    if source_format.lower() == "bgr":
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        return image  # Already RGB


def resize_image_if_needed(image: np.ndarray, 
                          target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """
    Resize image if target size is specified
    
    Args:
        image: Input image (H, W, C)
        target_size: Target (width, height) or None
    
    Returns:
        Resized image or original if target_size is None
    """
    if target_size is None:
        return image
    
    current_height, current_width = image.shape[:2]
    target_width, target_height = target_size
    
    if current_width != target_width or current_height != target_height:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    else:
        return image
