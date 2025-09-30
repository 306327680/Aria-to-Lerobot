"""
Utility functions for aria2lerobot conversion

Provides logging, validation, path handling, and progress tracking utilities.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
from tqdm import tqdm


def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> logging.Logger:
    """Setup logging configuration"""
    
    # Create logger
    logger = logging.getLogger("aria2lerobot")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def validate_paths(vrs_path: Path, mps_dir: Path, out_dir: Path, 
                   overwrite: bool = False) -> List[str]:
    """
    Validate input and output paths
    
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    # Check VRS file
    if not vrs_path.exists():
        errors.append(f"VRS file not found: {vrs_path}")
    elif not vrs_path.is_file():
        errors.append(f"VRS path is not a file: {vrs_path}")
    elif vrs_path.suffix.lower() != '.vrs':
        errors.append(f"VRS file should have .vrs extension: {vrs_path}")
    
    # Check MPS directory
    if not mps_dir.exists():
        errors.append(f"MPS directory not found: {mps_dir}")
    elif not mps_dir.is_dir():
        errors.append(f"MPS path is not a directory: {mps_dir}")
    else:
        # Check for essential MPS subdirectories
        essential_dirs = ["slam", "hand_tracking", "eye_gaze"]
        found_dirs = [d for d in essential_dirs if (mps_dir / d).exists()]
        if not found_dirs:
            errors.append(f"MPS directory missing essential subdirs: {essential_dirs}")
    
    # Check output directory
    if out_dir.exists():
        if not overwrite:
            errors.append(f"Output directory exists (use --overwrite): {out_dir}")
        elif not out_dir.is_dir():
            errors.append(f"Output path exists but is not a directory: {out_dir}")
    
    return errors


def create_output_structure(out_dir: Path, legacy_v21: bool = False) -> Dict[str, Path]:
    """
    Create output directory structure following standard LeRobot format
    
    Standard LeRobot structure:
    <dataset_name>/
    ├── data/chunk-000/episode_XXXXXX.parquet
    ├── videos/chunk-000/observation.images.main/episode_XXXXXX.mp4
    ├── meta/info.json, episodes.jsonl, tasks.jsonl, stats.json
    
    Returns:
        Dictionary mapping structure names to paths
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Standard LeRobot structure (same for v2.0, v2.1, v3.0 - only metadata differs)
    structure = {
        "root": out_dir,
        "meta": out_dir / "meta",
        "data": out_dir / "data" / "chunk-000",
        "videos": out_dir / "videos" / "chunk-000",
        "video_main": out_dir / "videos" / "chunk-000" / "observation.images.main",
    }
        
    # Create subdirectories
    for path in structure.values():
        if path != out_dir:
            path.mkdir(parents=True, exist_ok=True)
    
    return structure


def estimate_fps_from_timestamps(timestamps_ns: List[int]) -> float:
    """Estimate FPS from timestamp sequence"""
    if len(timestamps_ns) < 2:
        return 30.0  # Default fallback
    
    # Calculate intervals in seconds
    intervals_s = np.diff(timestamps_ns) / 1e9
    
    # Remove outliers (> 2x median)
    median_interval = np.median(intervals_s)
    clean_intervals = intervals_s[intervals_s <= 2 * median_interval]
    
    if len(clean_intervals) == 0:
        return 30.0
    
    # FPS = 1 / mean_interval
    mean_interval = np.mean(clean_intervals)
    fps = 1.0 / mean_interval if mean_interval > 0 else 30.0
    
    return min(fps, 120.0)  # Cap at 120 FPS


def compute_statistics(data: np.ndarray, name: str = "data") -> Dict[str, Any]:
    """Compute statistics for continuous data"""
    if len(data) == 0:
        return {"name": name, "count": 0}
    
    # Remove NaN values for statistics
    clean_data = data[~np.isnan(data)]
    nan_count = len(data) - len(clean_data)
    
    if len(clean_data) == 0:
        return {
            "name": name,
            "count": len(data),
            "valid_count": 0,
            "nan_count": nan_count,
            "nan_ratio": 1.0,
        }
    
    stats = {
        "name": name,
        "count": len(data),
        "valid_count": len(clean_data),
        "nan_count": nan_count,
        "nan_ratio": nan_count / len(data),
        "mean": float(np.mean(clean_data)),
        "std": float(np.std(clean_data)),
        "min": float(np.min(clean_data)),
        "max": float(np.max(clean_data)),
    }
    
    # Add percentiles
    if len(clean_data) >= 3:
        percentiles = np.percentile(clean_data, [25, 50, 75])
        stats.update({
            "q25": float(percentiles[0]),
            "median": float(percentiles[1]),
            "q75": float(percentiles[2]),
        })
    
    return stats


class ConversionProgress:
    """Progress tracking for conversion process"""
    
    def __init__(self, total_frames: int, log_interval: int = 100):
        self.total_frames = total_frames
        self.log_interval = log_interval
        self.start_time = time.time()
        self.processed_frames = 0
        self.valid_frames = 0
        self.dropped_frames = 0
        self.logger = logging.getLogger("aria2lerobot")
        
        # Progress bar
        self.pbar = tqdm(
            total=total_frames,
            desc="Converting frames",
            unit="frames",
            dynamic_ncols=True
        )
    
    def update(self, valid: bool = True):
        """Update progress counters"""
        self.processed_frames += 1
        if valid:
            self.valid_frames += 1
        else:
            self.dropped_frames += 1
        
        self.pbar.update(1)
        
        # Log periodically
        if self.processed_frames % self.log_interval == 0:
            self._log_progress()
    
    def _log_progress(self):
        """Log current progress"""
        elapsed = time.time() - self.start_time
        fps = self.processed_frames / elapsed if elapsed > 0 else 0
        
        self.logger.info(
            f"Progress: {self.processed_frames}/{self.total_frames} "
            f"({100 * self.processed_frames / self.total_frames:.1f}%) "
            f"- Valid: {self.valid_frames}, Dropped: {self.dropped_frames} "
            f"- Speed: {fps:.1f} fps"
        )
    
    def finish(self) -> Dict[str, Any]:
        """Finish progress tracking and return summary"""
        self.pbar.close()
        
        elapsed = time.time() - self.start_time
        avg_fps = self.processed_frames / elapsed if elapsed > 0 else 0
        
        summary = {
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "valid_frames": self.valid_frames,
            "dropped_frames": self.dropped_frames,
            "drop_rate": self.dropped_frames / self.processed_frames if self.processed_frames > 0 else 0,
            "elapsed_time_s": elapsed,
            "avg_fps": avg_fps,
        }
        
        self.logger.info(
            f"Conversion completed: {self.valid_frames}/{self.total_frames} valid frames "
            f"({100 * self.valid_frames / self.total_frames:.1f}%) "
            f"in {elapsed:.1f}s ({avg_fps:.1f} fps)"
        )
        
        return summary


def validate_conversion_output(out_dir: Path, expected_frames: int, 
                              legacy_v21: bool = False) -> List[str]:
    """
    Validate conversion output for consistency
    
    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []
    
    try:
        if legacy_v21:
            # v2.1 validation
            parquet_files = list(out_dir.glob("episode_*.parquet"))
            video_files = list(out_dir.glob("episode_*.mp4"))
            
            if not parquet_files:
                errors.append("No episode parquet files found")
            if not video_files:
                errors.append("No episode video files found")
            
            # Check file count consistency
            if len(parquet_files) != len(video_files):
                errors.append(f"Mismatch: {len(parquet_files)} parquet vs {len(video_files)} video files")
        
        else:
            # v3.0 validation
            meta_dir = out_dir / "meta"
            data_dir = out_dir / "data"
            videos_dir = out_dir / "videos"
            
            # Check directory structure
            for dir_path in [meta_dir, data_dir, videos_dir]:
                if not dir_path.exists():
                    errors.append(f"Missing directory: {dir_path}")
            
            # Check meta files
            info_file = meta_dir / "info.json"
            stats_file = meta_dir / "stats.json"
            
            if not info_file.exists():
                errors.append("Missing meta/info.json")
            if not stats_file.exists():
                errors.append("Missing meta/stats.json")
            
            # Check data files (in chunk subdirectories)
            parquet_files = list(data_dir.glob("chunk-*/*.parquet"))
            if not parquet_files:
                errors.append("No data parquet files found")

            # Check video files (in chunk subdirectories)
            video_files = list(videos_dir.glob("chunk-*/*/*.mp4"))
            if not video_files:
                errors.append("No video files found")
        
        # Validate frame counts (basic check)
        if parquet_files:
            try:
                import pyarrow.parquet as pq
                total_parquet_rows = 0
                for pq_file in parquet_files:
                    table = pq.read_table(pq_file)
                    total_parquet_rows += len(table)
                
                if abs(total_parquet_rows - expected_frames) > expected_frames * 0.05:  # 5% tolerance
                    errors.append(
                        f"Frame count mismatch: expected ~{expected_frames}, "
                        f"got {total_parquet_rows} in parquet files"
                    )
                    
            except Exception as e:
                errors.append(f"Failed to validate parquet frame counts: {e}")
    
    except Exception as e:
        errors.append(f"Validation error: {e}")
    
    return errors


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division with default value"""
    return numerator / denominator if denominator != 0 else default


def format_bytes(bytes_count: int) -> str:
    """Format bytes in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_count < 1024.0:
            return f"{bytes_count:.1f} {unit}"
        bytes_count /= 1024.0
    return f"{bytes_count:.1f} PB"


def format_duration(seconds: float) -> str:
    """Format duration in human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def check_memory_usage() -> Dict[str, float]:
    """Check current memory usage (if psutil available)"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "percent": process.memory_percent(),
        }
    except ImportError:
        return {"error": "psutil not available"}
    except Exception as e:
        return {"error": str(e)}


def cleanup_temp_files(temp_dir: Path):
    """Clean up temporary files"""
    if temp_dir.exists() and temp_dir.is_dir():
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except Exception as e:
            logging.getLogger("aria2lerobot").warning(f"Failed to cleanup {temp_dir}: {e}")
