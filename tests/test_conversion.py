"""
Integration tests for conversion functionality

Tests the main conversion pipeline with mock data and validation.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aria2lerobot.convert import (
    _extract_frame_data, _update_processing_stats, _finalize_processing_stats
)
from aria2lerobot.schema import LEROBOT_V3_SCHEMA, validate_schema_data


class MockAriaSession:
    """Mock AriaSession for testing"""
    
    def __init__(self):
        self.hand_projector = Mock()
        self.capabilities = Mock()
        self.capabilities.has_hands = True
        self.capabilities.has_trajectory = True
    
    def nearest_pose(self, timestamp_ns, max_slop_ns=None):
        """Mock pose data"""
        if timestamp_ns % 10000 == 0:  # Every 10000th timestamp has no pose (less frequent)
            return None
        
        # Create mock pose
        pose = Mock()
        pose.T_world_device = Mock()
        pose.T_world_device.to_matrix.return_value = np.eye(4, dtype=np.float32)
        return pose
    
    def nearest_hands(self, timestamp_ns, max_slop_ns=None):
        """Mock hand data"""
        if timestamp_ns % 5000 == 0:  # Every 5000th timestamp has no hands (less frequent)
            return None
        
        return Mock()  # Non-None indicates hands available
    
    def get_session_info(self):
        """Mock session info"""
        return {
            "capabilities": {
                "has_vrs": True,
                "has_trajectory": True,
                "has_hands": True,
            },
            "trajectory_time_range_ns": (1000000000, 2000000000),
        }


class TestFrameDataExtraction:
    """Test frame data extraction functionality"""
    
    def test_extract_frame_data_success(self):
        """Test successful frame data extraction"""
        session = MockAriaSession()
        
        # Mock landmarks_in_world method
        session.hand_projector.landmarks_in_world.return_value = {
            "left": np.random.rand(21, 3).astype(np.float32),
            "right": None,
        }
        
        timestamp_ns = 1234567890001  # Not divisible by 5000 or 10000
        step = 42
        episode_id = 0
        max_slop_ns = 5000000
        
        result = _extract_frame_data(session, timestamp_ns, step, episode_id, max_slop_ns)
        
        assert result is not None
        assert isinstance(result, dict)
        
        # Check required fields
        assert "timestamp" in result
        assert "timestamp_ns" in result
        assert "episode_id" in result
        assert "step" in result
        
        # Check values
        assert result["timestamp"] == timestamp_ns / 1e9
        assert result["timestamp_ns"] == timestamp_ns
        assert result["episode_id"] == episode_id
        assert result["step"] == step
        
        # Check pose data
        assert "status.pose.world_device" in result
        assert "status.pose.valid" in result
        assert result["status.pose.valid"] == True
        
        # Check hand data
        assert "status.hands.landmarks_world" in result
        assert "status.hands.valid_left" in result
        assert "status.hands.valid_right" in result
        
        # Check tactile placeholders
        assert "action.tactile.left.force_total" in result
        assert np.isnan(result["action.tactile.left.force_total"])
    
    def test_extract_frame_data_missing_pose(self):
        """Test frame data extraction with missing pose"""
        session = MockAriaSession()
        session.hand_projector.landmarks_in_world.return_value = {"left": None, "right": None}
        
        # Use timestamp that triggers missing pose
        timestamp_ns = 10000  # Divisible by 10000
        step = 0
        episode_id = 0
        max_slop_ns = 5000000
        
        result = _extract_frame_data(session, timestamp_ns, step, episode_id, max_slop_ns)
        
        assert result is not None
        assert result["status.pose.valid"] == False
        assert np.all(np.isnan(result["status.pose.world_device"]))
    
    def test_extract_frame_data_missing_hands(self):
        """Test frame data extraction with missing hands"""
        session = MockAriaSession()
        session.hand_projector.landmarks_in_world.return_value = {"left": None, "right": None}
        
        # Use timestamp that triggers missing hands
        timestamp_ns = 5000  # Divisible by 5000
        step = 0
        episode_id = 0
        max_slop_ns = 5000000
        
        result = _extract_frame_data(session, timestamp_ns, step, episode_id, max_slop_ns)
        
        assert result is not None
        assert result["status.hands.valid_left"] == False
        assert result["status.hands.valid_right"] == False
        assert np.all(np.isnan(result["status.hands.landmarks_world"]))
    
    def test_extract_frame_data_schema_compliance(self):
        """Test that extracted data complies with schema"""
        session = MockAriaSession()
        session.hand_projector.landmarks_in_world.return_value = {
            "left": np.random.rand(21, 3).astype(np.float32),
            "right": np.random.rand(21, 3).astype(np.float32),
        }
        
        timestamp_ns = 1234567890000
        step = 42
        episode_id = 0
        max_slop_ns = 5000000
        
        result = _extract_frame_data(session, timestamp_ns, step, episode_id, max_slop_ns)
        
        # Validate against schema
        schema = LEROBOT_V3_SCHEMA
        errors = validate_schema_data(result, schema)
        
        # Should have minimal errors (only missing non-required columns)
        missing_errors = [e for e in errors if "Missing required column" in e]
        assert len(missing_errors) == 0  # No missing required columns


class TestProcessingStats:
    """Test processing statistics functionality"""
    
    def test_update_processing_stats(self):
        """Test processing statistics updates"""
        processed_data = {
            "hand_stats": {"valid": 0, "total": 0},
            "pose_stats": {"valid": 0, "total": 0},
        }
        
        # Test with valid data
        row_data = {
            "status.hands.valid_left": True,
            "status.hands.valid_right": False,
            "status.pose.valid": True,
        }
        
        _update_processing_stats(row_data, processed_data)
        
        assert processed_data["hand_stats"]["total"] == 1
        assert processed_data["hand_stats"]["valid"] == 1
        assert processed_data["pose_stats"]["total"] == 1
        assert processed_data["pose_stats"]["valid"] == 1
        
        # Test with invalid data
        row_data_invalid = {
            "status.hands.valid_left": False,
            "status.hands.valid_right": False,
            "status.pose.valid": False,
        }
        
        _update_processing_stats(row_data_invalid, processed_data)
        
        assert processed_data["hand_stats"]["total"] == 2
        assert processed_data["hand_stats"]["valid"] == 1  # Still 1
        assert processed_data["pose_stats"]["total"] == 2
        assert processed_data["pose_stats"]["valid"] == 1  # Still 1
    
    def test_finalize_processing_stats(self):
        """Test processing statistics finalization"""
        processed_data = {
            "hand_stats": {"valid": 8, "total": 10},
            "pose_stats": {"valid": 9, "total": 10},
        }
        
        _finalize_processing_stats(processed_data)
        
        assert processed_data["hand_stats"]["valid_rate"] == 0.8
        assert processed_data["pose_stats"]["valid_rate"] == 0.9
    
    def test_finalize_processing_stats_zero_total(self):
        """Test processing statistics finalization with zero total"""
        processed_data = {
            "hand_stats": {"valid": 0, "total": 0},
            "pose_stats": {"valid": 0, "total": 0},
        }
        
        _finalize_processing_stats(processed_data)
        
        assert processed_data["hand_stats"]["valid_rate"] == 0.0
        assert processed_data["pose_stats"]["valid_rate"] == 0.0


class TestConversionIntegration:
    """Integration tests for the conversion process"""
    
    @patch('aria2lerobot.convert.AriaSession')
    def test_conversion_pipeline_structure(self, mock_aria_session_class):
        """Test the overall conversion pipeline structure"""
        # This is a structural test - we're not running the full conversion
        # but testing that the main components can be imported and called
        
        from aria2lerobot.convert import convert
        
        # The convert function should be importable and callable
        assert callable(convert)
        
        # Test that required parameters are handled
        with pytest.raises((TypeError, ValueError)):
            # Should fail due to missing required parameters
            convert()
    
    def test_mock_session_behavior(self):
        """Test that our mock session behaves correctly"""
        session = MockAriaSession()
        
        # Test pose behavior
        pose1 = session.nearest_pose(1234)  # Should return pose
        assert pose1 is not None
        
        pose2 = session.nearest_pose(10000)  # Should return None (divisible by 10000)
        assert pose2 is None
        
        # Test hands behavior
        hands1 = session.nearest_hands(1234)  # Should return hands
        assert hands1 is not None
        
        hands2 = session.nearest_hands(5000)  # Should return None (divisible by 5000)
        assert hands2 is None
        
        # Test session info
        info = session.get_session_info()
        assert "capabilities" in info
        assert "trajectory_time_range_ns" in info


class TestDataIntegrity:
    """Test data integrity and consistency"""
    
    def test_hand_landmarks_consistency(self):
        """Test that hand landmarks maintain consistency"""
        session = MockAriaSession()
        
        # Create consistent mock data
        left_landmarks = np.random.rand(21, 3).astype(np.float32)
        session.hand_projector.landmarks_in_world.return_value = {
            "left": left_landmarks,
            "right": None,
        }
        
        result = _extract_frame_data(session, 1234567890001, 0, 0, 5000000)
        
        # Extract and validate hand data
        flattened_landmarks = result["status.hands.landmarks_world"]
        reconstructed = flattened_landmarks.reshape(21, 3)
        
        np.testing.assert_array_equal(reconstructed, left_landmarks)
        assert result["status.hands.valid_left"] == True
        assert result["status.hands.valid_right"] == False
    
    def test_pose_matrix_consistency(self):
        """Test that pose matrices maintain consistency"""
        session = MockAriaSession()
        session.hand_projector.landmarks_in_world.return_value = {"left": None, "right": None}
        
        # The mock returns identity matrix
        expected_matrix = np.eye(4, dtype=np.float32)
        
        result = _extract_frame_data(session, 1234567890001, 0, 0, 5000000)
        
        # Extract and validate pose data
        flattened_pose = result["status.pose.world_device"]
        reconstructed_pose = flattened_pose.reshape(4, 4)
        
        np.testing.assert_array_equal(reconstructed_pose, expected_matrix)
        assert result["status.pose.valid"] == True
    
    def test_timestamp_consistency(self):
        """Test that timestamps are handled consistently"""
        session = MockAriaSession()
        session.hand_projector.landmarks_in_world.return_value = {"left": None, "right": None}
        
        timestamp_ns = 1234567890123456
        step = 42
        episode_id = 1
        
        result = _extract_frame_data(session, timestamp_ns, step, episode_id, 5000000)
        
        assert result["timestamp_ns"] == timestamp_ns
        assert result["timestamp"] == timestamp_ns / 1e9
        assert result["step"] == step
        assert result["episode_id"] == episode_id
    
    def test_nan_placeholder_consistency(self):
        """Test that NaN placeholders are consistent"""
        session = MockAriaSession()
        session.hand_projector.landmarks_in_world.return_value = {"left": None, "right": None}
        
        result = _extract_frame_data(session, 1234567890001, 0, 0, 5000000)
        
        # Check tactile placeholders are NaN
        assert np.isnan(result["action.tactile.left.force_total"])
        assert np.isnan(result["action.tactile.right.force_total"])
        assert np.all(np.isnan(result["action.tactile.left.force_xyz"]))
        assert np.all(np.isnan(result["action.tactile.right.force_xyz"]))
        
        # Check masks are False
        assert result["action.mask.tactile.left.force_total"] == False
        assert result["action.mask.tactile.right.force_total"] == False
        assert np.all(result["action.mask.tactile.left.force_xyz"] == False)
        assert np.all(result["action.mask.tactile.right.force_xyz"] == False)


if __name__ == "__main__":
    pytest.main([__file__])
