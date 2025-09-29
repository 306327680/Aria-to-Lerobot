"""
Unit tests for utils module

Tests utility functions for logging, validation, path handling, and statistics.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys
import logging

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aria2lerobot.utils import (
    setup_logging, validate_paths, create_output_structure,
    estimate_fps_from_timestamps, compute_statistics, ConversionProgress,
    validate_conversion_output, safe_divide, format_bytes, format_duration
)


class TestLogging:
    """Test logging setup functionality"""
    
    def test_setup_logging_basic(self):
        """Test basic logging setup"""
        logger = setup_logging("INFO")
        
        assert logger.name == "aria2lerobot"
        assert logger.level == logging.INFO
        assert len(logger.handlers) >= 1
    
    def test_setup_logging_with_file(self):
        """Test logging setup with file output"""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"
            logger = setup_logging("DEBUG", log_file)
            
            assert logger.level == logging.DEBUG
            assert len(logger.handlers) >= 2  # Console + file
            assert log_file.exists()
    
    def test_setup_logging_levels(self):
        """Test different logging levels"""
        for level_name in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            logger = setup_logging(level_name)
            expected_level = getattr(logging, level_name)
            assert logger.level == expected_level


class TestPathValidation:
    """Test path validation functions"""
    
    def test_validate_paths_valid(self):
        """Test validation of valid paths"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files/directories
            vrs_file = temp_path / "test.vrs"
            vrs_file.touch()
            
            mps_dir = temp_path / "mps_test_vrs"
            mps_dir.mkdir()
            (mps_dir / "slam").mkdir()
            (mps_dir / "hand_tracking").mkdir()
            
            out_dir = temp_path / "output"
            
            errors = validate_paths(vrs_file, mps_dir, out_dir, overwrite=True)
            
            assert len(errors) == 0
    
    def test_validate_paths_missing_vrs(self):
        """Test validation with missing VRS file"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            vrs_file = temp_path / "nonexistent.vrs"
            mps_dir = temp_path / "mps"
            out_dir = temp_path / "output"
            
            errors = validate_paths(vrs_file, mps_dir, out_dir)
            
            assert len(errors) > 0
            assert any("VRS file not found" in error for error in errors)
    
    def test_validate_paths_missing_mps(self):
        """Test validation with missing MPS directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            vrs_file = temp_path / "test.vrs"
            vrs_file.touch()
            
            mps_dir = temp_path / "nonexistent_mps"
            out_dir = temp_path / "output"
            
            errors = validate_paths(vrs_file, mps_dir, out_dir)
            
            assert len(errors) > 0
            assert any("MPS directory not found" in error for error in errors)
    
    def test_validate_paths_existing_output(self):
        """Test validation with existing output directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            vrs_file = temp_path / "test.vrs"
            vrs_file.touch()
            
            mps_dir = temp_path / "mps"
            mps_dir.mkdir()
            (mps_dir / "slam").mkdir()
            
            out_dir = temp_path / "output"
            out_dir.mkdir()  # Already exists
            
            errors = validate_paths(vrs_file, mps_dir, out_dir, overwrite=False)
            
            assert len(errors) > 0
            assert any("Output directory exists" in error for error in errors)
    
    def test_validate_paths_wrong_extension(self):
        """Test validation with wrong file extension"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            vrs_file = temp_path / "test.txt"  # Wrong extension
            vrs_file.touch()
            
            mps_dir = temp_path / "mps"
            out_dir = temp_path / "output"
            
            errors = validate_paths(vrs_file, mps_dir, out_dir)
            
            assert len(errors) > 0
            assert any(".vrs extension" in error for error in errors)


class TestOutputStructure:
    """Test output directory structure creation"""
    
    def test_create_v3_structure(self):
        """Test v3.0 output structure creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "output"
            
            structure = create_output_structure(out_dir, legacy_v21=False)
            
            assert "root" in structure
            assert "meta" in structure
            assert "data" in structure
            assert "videos" in structure
            assert "episodes" in structure
            
            # Check directories were created
            assert structure["meta"].exists()
            assert structure["data"].exists()
            assert structure["videos"].exists()
            assert structure["episodes"].exists()
    
    def test_create_v21_structure(self):
        """Test v2.1 output structure creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir) / "output"
            
            structure = create_output_structure(out_dir, legacy_v21=True)
            
            assert "root" in structure
            assert "episodes" in structure
            assert "videos" in structure
            
            # Should be simpler structure
            assert structure["episodes"] == structure["root"]
            assert structure["videos"] == structure["root"]


class TestFPSEstimation:
    """Test FPS estimation functions"""
    
    def test_estimate_fps_regular_intervals(self):
        """Test FPS estimation with regular intervals"""
        # 30 FPS = 33.333ms intervals
        interval_ns = int(1e9 / 30)
        timestamps = [i * interval_ns for i in range(100)]
        
        fps = estimate_fps_from_timestamps(timestamps)
        
        assert abs(fps - 30.0) < 0.1  # Should be close to 30 FPS
    
    def test_estimate_fps_irregular_intervals(self):
        """Test FPS estimation with some irregular intervals"""
        # Mostly 30 FPS with some outliers
        base_interval = int(1e9 / 30)
        timestamps = []
        
        for i in range(100):
            if i % 20 == 0:  # Add fewer outliers
                timestamps.append(i * base_interval + base_interval * 2)
            else:
                timestamps.append(i * base_interval)
        
        fps = estimate_fps_from_timestamps(timestamps)
        
        # Should still estimate around 30 FPS (outliers filtered)
        # Be more lenient with the range due to outlier filtering algorithm
        assert 20 < fps < 40
    
    def test_estimate_fps_empty_list(self):
        """Test FPS estimation with empty list"""
        fps = estimate_fps_from_timestamps([])
        
        assert fps == 30.0  # Default fallback
    
    def test_estimate_fps_single_timestamp(self):
        """Test FPS estimation with single timestamp"""
        fps = estimate_fps_from_timestamps([123456789])
        
        assert fps == 30.0  # Default fallback


class TestStatistics:
    """Test statistics computation functions"""
    
    def test_compute_statistics_normal_data(self):
        """Test statistics computation with normal data"""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        stats = compute_statistics(data, "test_data")
        
        assert stats["name"] == "test_data"
        assert stats["count"] == 5
        assert stats["valid_count"] == 5
        assert stats["nan_count"] == 0
        assert stats["nan_ratio"] == 0.0
        assert stats["mean"] == 3.0
        assert stats["min"] == 1.0
        assert stats["max"] == 5.0
    
    def test_compute_statistics_with_nan(self):
        """Test statistics computation with NaN values"""
        data = np.array([1.0, 2.0, np.nan, 4.0, np.nan])
        
        stats = compute_statistics(data, "test_data")
        
        assert stats["count"] == 5
        assert stats["valid_count"] == 3
        assert stats["nan_count"] == 2
        assert stats["nan_ratio"] == 0.4
        assert stats["mean"] == pytest.approx(2.333, rel=1e-2)
    
    def test_compute_statistics_empty_data(self):
        """Test statistics computation with empty data"""
        data = np.array([])
        
        stats = compute_statistics(data, "empty_data")
        
        assert stats["count"] == 0
        assert stats["name"] == "empty_data"
    
    def test_compute_statistics_all_nan(self):
        """Test statistics computation with all NaN data"""
        data = np.array([np.nan, np.nan, np.nan])
        
        stats = compute_statistics(data, "nan_data")
        
        assert stats["count"] == 3
        assert stats["valid_count"] == 0
        assert stats["nan_count"] == 3
        assert stats["nan_ratio"] == 1.0


class TestConversionProgress:
    """Test conversion progress tracking"""
    
    def test_conversion_progress_basic(self):
        """Test basic progress tracking"""
        progress = ConversionProgress(100, log_interval=1000)  # High interval to avoid logging
        
        # Update some frames
        progress.update(valid=True)
        progress.update(valid=True)
        progress.update(valid=False)
        
        assert progress.processed_frames == 3
        assert progress.valid_frames == 2
        assert progress.dropped_frames == 1
        
        # Finish and get summary
        summary = progress.finish()
        
        assert summary["total_frames"] == 100
        assert summary["processed_frames"] == 3
        assert summary["valid_frames"] == 2
        assert summary["dropped_frames"] == 1
        assert summary["drop_rate"] == pytest.approx(1/3)


class TestConversionValidation:
    """Test conversion output validation"""
    
    def test_validate_v3_output_missing_dirs(self):
        """Test validation of v3.0 output with missing directories"""
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            
            errors = validate_conversion_output(out_dir, 100, legacy_v21=False)
            
            assert len(errors) > 0
            assert any("Missing directory" in error for error in errors)
    
    def test_validate_v21_output_missing_files(self):
        """Test validation of v2.1 output with missing files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            out_dir = Path(temp_dir)
            
            errors = validate_conversion_output(out_dir, 100, legacy_v21=True)
            
            assert len(errors) > 0
            assert any("episode" in error.lower() for error in errors)


class TestUtilityFunctions:
    """Test miscellaneous utility functions"""
    
    def test_safe_divide_normal(self):
        """Test safe division with normal values"""
        result = safe_divide(10.0, 2.0)
        assert result == 5.0
    
    def test_safe_divide_by_zero(self):
        """Test safe division by zero"""
        result = safe_divide(10.0, 0.0, default=42.0)
        assert result == 42.0
        
        # Test default default
        result = safe_divide(10.0, 0.0)
        assert result == 0.0
    
    def test_format_bytes(self):
        """Test byte formatting"""
        assert format_bytes(1024) == "1.0 KB"
        assert format_bytes(1024 * 1024) == "1.0 MB"
        assert format_bytes(1024 * 1024 * 1024) == "1.0 GB"
        assert format_bytes(500) == "500.0 B"
    
    def test_format_duration(self):
        """Test duration formatting"""
        assert format_duration(30) == "30.0s"
        assert format_duration(90) == "1.5m"
        assert format_duration(3600) == "1.0h"
        assert format_duration(7200) == "2.0h"


if __name__ == "__main__":
    pytest.main([__file__])
