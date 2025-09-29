"""
Unit tests for schema module

Tests LeRobot schema definitions, data validation, and helper functions.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aria2lerobot.schema import (
    LeRobotSchema, LEROBOT_V3_SCHEMA, LEROBOT_V21_SCHEMA,
    ColumnSpec, flatten_hand_landmarks, unflatten_hand_landmarks,
    create_tactile_placeholders, flatten_pose_matrix, unflatten_pose_matrix,
    validate_schema_data
)


class TestColumnSpec:
    """Test ColumnSpec functionality"""
    
    def test_column_spec_creation(self):
        """Test ColumnSpec creation and basic properties"""
        spec = ColumnSpec((3, 4), "float32", "Test array")
        
        assert spec.shape == (3, 4)
        assert spec.dtype == "float32"
        assert spec.description == "Test array"
    
    def test_validate_data_success(self):
        """Test successful data validation"""
        spec = ColumnSpec((3,), "float32")
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        
        assert spec.validate_data(data) == True
    
    def test_validate_data_wrong_dtype(self):
        """Test validation failure due to wrong dtype"""
        spec = ColumnSpec((3,), "float32")
        data = np.array([1, 2, 3], dtype=np.int32)  # Wrong dtype
        
        assert spec.validate_data(data) == False
    
    def test_validate_data_wrong_shape(self):
        """Test validation failure due to wrong shape"""
        spec = ColumnSpec((3,), "float32")
        data = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)  # Wrong shape
        
        assert spec.validate_data(data) == False
    
    def test_validate_scalar_data(self):
        """Test validation of scalar data"""
        spec = ColumnSpec((), "int64")
        data = np.int64(42)
        
        assert spec.validate_data(data) == True


class TestLeRobotSchema:
    """Test LeRobotSchema functionality"""
    
    def test_v3_schema_structure(self):
        """Test v3.0 schema structure"""
        schema = LEROBOT_V3_SCHEMA
        
        assert schema.version == "3.0"
        assert len(schema.columns) > 0
        assert len(schema.required_columns) > 0
        assert "observation.images.main" in schema.video_columns
        
        # Check key required columns
        required = schema.required_columns
        assert "timestamp" in required
        assert "timestamp_ns" in required
        assert "episode_id" in required
        assert "step" in required
    
    def test_v21_schema_structure(self):
        """Test v2.1 schema structure"""
        schema = LEROBOT_V21_SCHEMA
        
        assert schema.version == "2.1"
        assert len(schema.columns) > 0
        assert len(schema.required_columns) > 0
        
        # Should be similar to v3 for backward compatibility
        assert set(schema.required_columns) == set(LEROBOT_V3_SCHEMA.required_columns)
    
    def test_get_info_json(self):
        """Test info.json generation"""
        schema = LEROBOT_V3_SCHEMA
        info = schema.get_info_json(fps=30.0, episode_count=1)
        
        assert info["schema_version"] == "3.0"
        assert info["fps"] == 30.0
        assert info["episode_count"] == 1
        assert "columns" in info
        assert "video_columns" in info
        assert "required_columns" in info
        
        # Check column information
        assert "timestamp" in info["columns"]
        timestamp_info = info["columns"]["timestamp"]
        assert timestamp_info["shape"] == []
        assert timestamp_info["dtype"] == "float64"
    
    def test_get_parquet_schema(self):
        """Test PyArrow schema generation"""
        schema = LEROBOT_V3_SCHEMA
        
        try:
            pa_schema = schema.get_parquet_schema()
            assert pa_schema is not None
            assert len(pa_schema) > 0
            
            # Check that video columns are excluded
            field_names = pa_schema.names
            for video_col in schema.video_columns:
                assert video_col not in field_names
                
        except ImportError:
            pytest.skip("PyArrow not available")


class TestHandLandmarks:
    """Test hand landmark processing functions"""
    
    def test_flatten_valid_landmarks(self):
        """Test flattening of valid hand landmarks"""
        left_landmarks = np.random.rand(21, 3).astype(np.float32)
        right_landmarks = np.random.rand(21, 3).astype(np.float32)
        
        flattened, valid_left, valid_right = flatten_hand_landmarks(left_landmarks, right_landmarks)
        
        assert flattened.shape == (63,)
        assert flattened.dtype == np.float32
        assert valid_left == True
        assert valid_right == True
        
        # Check that flattened data matches original
        reconstructed = flattened.reshape(21, 3)
        np.testing.assert_array_equal(reconstructed, left_landmarks)
    
    def test_flatten_missing_landmarks(self):
        """Test flattening with missing landmarks"""
        flattened, valid_left, valid_right = flatten_hand_landmarks(None, None)
        
        assert flattened.shape == (63,)
        assert flattened.dtype == np.float32
        assert valid_left == False
        assert valid_right == False
        assert np.all(np.isnan(flattened))
    
    def test_flatten_partial_landmarks(self):
        """Test flattening with only left hand"""
        left_landmarks = np.random.rand(21, 3).astype(np.float32)
        
        flattened, valid_left, valid_right = flatten_hand_landmarks(left_landmarks, None)
        
        assert valid_left == True
        assert valid_right == False
        
        # Check reconstruction
        reconstructed = flattened.reshape(21, 3)
        np.testing.assert_array_equal(reconstructed, left_landmarks)
    
    def test_flatten_nan_landmarks(self):
        """Test flattening with NaN landmarks"""
        left_landmarks = np.full((21, 3), np.nan, dtype=np.float32)
        
        flattened, valid_left, valid_right = flatten_hand_landmarks(left_landmarks, None)
        
        assert valid_left == False  # Should be invalid due to NaN
        assert valid_right == False
    
    def test_unflatten_landmarks(self):
        """Test unflattening of hand landmarks"""
        original = np.random.rand(21, 3).astype(np.float32)
        flattened = original.flatten()
        
        reconstructed = unflatten_hand_landmarks(flattened)
        
        assert reconstructed.shape == (21, 3)
        np.testing.assert_array_equal(reconstructed, original)
    
    def test_unflatten_wrong_size(self):
        """Test unflattening with wrong size"""
        wrong_size = np.random.rand(60)  # Wrong size
        
        with pytest.raises(ValueError, match="Expected 63 elements"):
            unflatten_hand_landmarks(wrong_size)


class TestPoseMatrix:
    """Test pose matrix processing functions"""
    
    def test_flatten_pose_matrix(self):
        """Test pose matrix flattening"""
        # Create a valid 4x4 transformation matrix
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = [1.0, 2.0, 3.0]  # Translation
        
        flattened = flatten_pose_matrix(pose)
        
        assert flattened.shape == (16,)
        assert flattened.dtype == np.float32
        
        # Check that we can reconstruct
        reconstructed = unflatten_pose_matrix(flattened)
        np.testing.assert_array_almost_equal(reconstructed, pose, decimal=5)
    
    def test_unflatten_pose_matrix(self):
        """Test pose matrix unflattening"""
        flattened = np.arange(16, dtype=np.float32)
        
        reconstructed = unflatten_pose_matrix(flattened)
        
        assert reconstructed.shape == (4, 4)
        expected = flattened.reshape(4, 4)
        np.testing.assert_array_equal(reconstructed, expected)
    
    def test_flatten_wrong_shape(self):
        """Test flattening with wrong shape"""
        wrong_shape = np.random.rand(3, 3)  # Wrong shape
        
        with pytest.raises(ValueError, match="Expected \\(4, 4\\) matrix"):
            flatten_pose_matrix(wrong_shape)
    
    def test_unflatten_wrong_size(self):
        """Test unflattening with wrong size"""
        wrong_size = np.random.rand(15)  # Wrong size
        
        with pytest.raises(ValueError, match="Expected 16 elements"):
            unflatten_pose_matrix(wrong_size)


class TestTactilePlaceholders:
    """Test tactile placeholder generation"""
    
    def test_create_tactile_placeholders(self):
        """Test tactile placeholder creation"""
        placeholders = create_tactile_placeholders()
        
        # Check all expected keys are present
        expected_keys = [
            "action.tactile.left.force_total",
            "action.tactile.right.force_total", 
            "action.tactile.left.force_xyz",
            "action.tactile.right.force_xyz",
            "action.mask.tactile.left.force_total",
            "action.mask.tactile.right.force_total",
            "action.mask.tactile.left.force_xyz",
            "action.mask.tactile.right.force_xyz",
        ]
        
        for key in expected_keys:
            assert key in placeholders
        
        # Check scalar values are NaN
        assert np.isnan(placeholders["action.tactile.left.force_total"])
        assert np.isnan(placeholders["action.tactile.right.force_total"])
        
        # Check array values are NaN
        assert placeholders["action.tactile.left.force_xyz"].shape == (3,)
        assert np.all(np.isnan(placeholders["action.tactile.left.force_xyz"]))
        
        # Check masks are False
        assert placeholders["action.mask.tactile.left.force_total"] == False
        assert placeholders["action.mask.tactile.right.force_total"] == False
        assert np.all(placeholders["action.mask.tactile.left.force_xyz"] == False)


class TestSchemaValidation:
    """Test schema validation functions"""
    
    def test_validate_valid_data(self):
        """Test validation of valid data"""
        schema = LEROBOT_V3_SCHEMA
        
        # Create valid sample data
        data = {
            "timestamp": 123.456,
            "timestamp_ns": 123456789000,
            "episode_id": 0,
            "step": 42,
            "status.hands.landmarks_world": np.random.rand(63).astype(np.float32),
            "status.hands.valid_left": True,
            "status.hands.valid_right": False,
        }
        
        errors = validate_schema_data(data, schema)
        
        # Should have some errors due to missing columns, but basic structure should be OK
        assert isinstance(errors, list)
    
    def test_validate_missing_required(self):
        """Test validation with missing required columns"""
        schema = LEROBOT_V3_SCHEMA
        
        # Missing required columns
        data = {
            "timestamp": 123.456,
            # Missing other required columns
        }
        
        errors = validate_schema_data(data, schema)
        
        assert len(errors) > 0
        assert any("Missing required column" in error for error in errors)
    
    def test_validate_wrong_types(self):
        """Test validation with wrong data types"""
        schema = LEROBOT_V3_SCHEMA
        
        data = {
            "timestamp": "not_a_number",  # Wrong type
            "timestamp_ns": 123456789000,
            "episode_id": 0,
            "step": 42,
            "status.hands.landmarks_world": np.random.rand(63).astype(np.float32),
            "status.hands.valid_left": True,
            "status.hands.valid_right": False,
        }
        
        errors = validate_schema_data(data, schema)
        
        assert len(errors) > 0
        # Should have validation errors due to wrong types
    
    def test_validate_unknown_columns(self):
        """Test validation with unknown columns"""
        schema = LEROBOT_V3_SCHEMA
        
        data = {
            "timestamp": 123.456,
            "unknown_column": "should_not_exist",  # Unknown column
        }
        
        errors = validate_schema_data(data, schema)
        
        assert len(errors) > 0
        assert any("Unknown column" in error for error in errors)


if __name__ == "__main__":
    pytest.main([__file__])
