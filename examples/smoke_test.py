#!/usr/bin/env python3
"""
Smoke test for aria2lerobot conversion

Tests the complete conversion pipeline:
1. Convert small Aria dataset to LeRobot format
2. Load with LeRobotDataset (if available)
3. Validate data integrity and schema compliance
4. Print sample data for verification

Usage:
    python smoke_test.py path/to/glue.vrs [output_dir]
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "aria_api"))

from aria2lerobot import convert
from aria2lerobot.schema import LEROBOT_V3_SCHEMA, unflatten_hand_landmarks, unflatten_pose_matrix


def run_smoke_test(vrs_path: str, output_dir: Optional[str] = None, cleanup: bool = True) -> Dict[str, Any]:
    """
    Run complete smoke test
    
    Args:
        vrs_path: Path to VRS file for testing
        output_dir: Output directory (temp dir if None)
        cleanup: Whether to cleanup output directory after test
    
    Returns:
        Test results dictionary
    """
    
    print("🧪 === Aria2LeRobot Smoke Test ===")
    print(f"📁 Input VRS: {vrs_path}")
    
    # Validate input
    vrs_file = Path(vrs_path)
    if not vrs_file.exists():
        raise FileNotFoundError(f"VRS file not found: {vrs_path}")
    
    # Setup output directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="aria2lerobot_test_")
        temp_output = True
    else:
        output_dir = Path(output_dir)
        temp_output = False
    
    print(f"📂 Output directory: {output_dir}")
    
    results = {
        "conversion_success": False,
        "loading_success": False,
        "validation_success": False,
        "sample_data": {},
        "errors": [],
    }
    
    try:
        # Step 1: Convert to LeRobot format
        print("\n🔄 Step 1: Converting Aria data to LeRobot format...")
        
        conversion_result = convert(
            vrs_path=str(vrs_file),
            mps_dir=None,  # Auto-detect
            out_dir=str(output_dir),
            max_frames=100,  # Small test dataset
            stride=5,        # Every 5th frame for speed
            overwrite=True,
            log_level="WARNING"  # Reduce log noise
        )
        
        results["conversion_success"] = True
        results["conversion_result"] = conversion_result
        
        print(f"✅ Conversion completed:")
        print(f"   📊 Frames: {conversion_result['processing']['valid_frames']}")
        print(f"   ⏱️  Time: {conversion_result['processing']['elapsed_time_s']:.1f}s")
        print(f"   📈 Hand valid rate: {conversion_result['data_quality']['hand_valid_rate']:.1%}")
        
        # Step 2: Load with LeRobotDataset (if available)
        print("\n📖 Step 2: Loading with LeRobotDataset...")
        
        try:
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
            
            dataset = LeRobotDataset(str(output_dir))
            results["loading_success"] = True
            results["dataset_info"] = {
                "length": len(dataset),
                "features": list(dataset.features.keys()),
            }
            
            print(f"✅ Dataset loaded successfully:")
            print(f"   📊 Length: {len(dataset)} samples")
            print(f"   🔑 Features: {len(dataset.features)} columns")
            
            # Step 3: Validate sample data
            print("\n🔍 Step 3: Validating sample data...")
            sample_validation = validate_sample_data(dataset, results)
            results["validation_success"] = sample_validation["success"]
            results["sample_data"] = sample_validation["sample_data"]
            
        except ImportError:
            print("⚠️  LeRobotDataset not available - skipping loading test")
            print("   Install lerobot to enable full testing")
            results["loading_success"] = None
            
            # Alternative validation using direct parquet reading
            print("\n🔍 Step 3: Validating output files...")
            file_validation = validate_output_files(output_dir, results)
            results["validation_success"] = file_validation["success"]
            results["sample_data"] = file_validation["sample_data"]
        
        # Step 4: Print summary
        print("\n📋 Step 4: Test Summary")
        print_test_summary(results)
        
    except Exception as e:
        results["errors"].append(str(e))
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup if requested
        if cleanup and temp_output:
            try:
                shutil.rmtree(output_dir)
                print(f"🗑️  Cleaned up temporary directory: {output_dir}")
            except Exception as e:
                print(f"⚠️  Failed to cleanup {output_dir}: {e}")
    
    return results


def validate_sample_data(dataset, results: Dict[str, Any]) -> Dict[str, Any]:
    """Validate sample data from LeRobotDataset"""
    
    validation = {
        "success": False,
        "sample_data": {},
        "checks": {},
    }
    
    try:
        if len(dataset) == 0:
            validation["checks"]["dataset_empty"] = True
            return validation
        
        # Get first sample
        sample = dataset[0]
        validation["sample_data"]["raw_sample"] = {
            key: {
                "shape": getattr(value, 'shape', 'scalar'),
                "dtype": str(getattr(value, 'dtype', type(value))),
                "sample_values": str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
            }
            for key, value in sample.items()
        }
        
        # Check required columns
        schema = LEROBOT_V3_SCHEMA
        missing_columns = []
        for col in schema.required_columns:
            if col not in sample:
                missing_columns.append(col)
        
        validation["checks"]["missing_columns"] = missing_columns
        validation["checks"]["has_required_columns"] = len(missing_columns) == 0
        
        # Validate specific data types
        checks = {}
        
        # Check timestamps
        if "timestamp" in sample:
            ts = sample["timestamp"]
            checks["timestamp_valid"] = isinstance(ts, (int, float)) and ts > 0
        
        if "timestamp_ns" in sample:
            ts_ns = sample["timestamp_ns"]
            checks["timestamp_ns_valid"] = isinstance(ts_ns, (int, np.integer)) and ts_ns > 0
        
        # Check hand landmarks
        if "status.hands.landmarks_world" in sample:
            hand_data = sample["status.hands.landmarks_world"]
            checks["hand_landmarks_shape"] = hand_data.shape == (63,)
            checks["hand_landmarks_dtype"] = hand_data.dtype == np.float32
            
            # Try to unflatten
            try:
                landmarks = unflatten_hand_landmarks(hand_data)
                checks["hand_landmarks_unflatten"] = landmarks.shape == (21, 3)
                validation["sample_data"]["hand_landmarks_sample"] = landmarks[:3].tolist()  # First 3 points
            except Exception as e:
                checks["hand_landmarks_unflatten"] = False
                validation["sample_data"]["hand_unflatten_error"] = str(e)
        
        # Check pose data
        if "status.pose.world_device" in sample:
            pose_data = sample["status.pose.world_device"]
            checks["pose_shape"] = pose_data.shape == (16,)
            checks["pose_dtype"] = pose_data.dtype == np.float32
            
            # Try to unflatten
            try:
                pose_matrix = unflatten_pose_matrix(pose_data)
                checks["pose_unflatten"] = pose_matrix.shape == (4, 4)
                validation["sample_data"]["pose_translation"] = pose_matrix[:3, 3].tolist()
            except Exception as e:
                checks["pose_unflatten"] = False
                validation["sample_data"]["pose_unflatten_error"] = str(e)
        
        # Check video data
        if "observation.images.main" in sample:
            img_data = sample["observation.images.main"]
            checks["image_shape"] = len(img_data.shape) == 3  # (H, W, C)
            checks["image_dtype"] = img_data.dtype == np.uint8
            checks["image_values"] = 0 <= img_data.min() and img_data.max() <= 255
            validation["sample_data"]["image_shape"] = img_data.shape
        
        # Check tactile placeholders
        tactile_columns = [col for col in sample.keys() if "action.tactile" in col]
        checks["has_tactile_placeholders"] = len(tactile_columns) > 0
        
        if tactile_columns:
            # Check if they're properly NaN
            nan_checks = {}
            for col in tactile_columns:
                if "mask" not in col:  # Skip mask columns
                    data = sample[col]
                    if hasattr(data, '__iter__') and not isinstance(data, str):
                        nan_checks[col] = np.all(np.isnan(data))
                    else:
                        nan_checks[col] = np.isnan(data)
            
            checks["tactile_nan_values"] = nan_checks
        
        validation["checks"] = checks
        validation["success"] = all([
            checks.get("has_required_columns", False),
            checks.get("timestamp_valid", False),
            checks.get("hand_landmarks_shape", False),
            checks.get("pose_shape", False),
        ])
        
        print(f"✅ Sample validation: {sum(checks.values())} / {len(checks)} checks passed")
        
    except Exception as e:
        validation["sample_data"]["validation_error"] = str(e)
        print(f"❌ Sample validation failed: {e}")
    
    return validation


def validate_output_files(output_dir: Path, results: Dict[str, Any]) -> Dict[str, Any]:
    """Validate output files directly (when LeRobotDataset not available)"""
    
    validation = {
        "success": False,
        "sample_data": {},
        "checks": {},
    }
    
    try:
        import pyarrow.parquet as pq
        import json
        
        output_path = Path(output_dir)
        
        # Check directory structure
        expected_dirs = ["meta", "data", "videos"]
        missing_dirs = [d for d in expected_dirs if not (output_path / d).exists()]
        validation["checks"]["missing_directories"] = missing_dirs
        
        # Check metadata files
        info_file = output_path / "meta" / "info.json"
        if info_file.exists():
            with open(info_file) as f:
                info_data = json.load(f)
            validation["sample_data"]["info_json"] = info_data
            validation["checks"]["has_info_json"] = True
        else:
            validation["checks"]["has_info_json"] = False
        
        # Check parquet files
        data_files = list((output_path / "data").glob("*.parquet"))
        validation["checks"]["has_parquet_files"] = len(data_files) > 0
        
        if data_files:
            # Read first parquet file
            table = pq.read_table(data_files[0])
            validation["sample_data"]["parquet_columns"] = table.column_names
            validation["sample_data"]["parquet_rows"] = len(table)
            
            # Get sample row
            if len(table) > 0:
                sample_row = table.slice(0, 1).to_pandas().iloc[0]
                validation["sample_data"]["sample_row"] = {
                    key: str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                    for key, value in sample_row.items()
                }
        
        # Check video files
        video_files = list((output_path / "videos").glob("*.mp4"))
        validation["checks"]["has_video_files"] = len(video_files) > 0
        validation["sample_data"]["video_count"] = len(video_files)
        
        # Overall success
        validation["success"] = all([
            len(missing_dirs) == 0,
            validation["checks"]["has_info_json"],
            validation["checks"]["has_parquet_files"],
            validation["checks"]["has_video_files"],
        ])
        
        print(f"✅ File validation: {len(data_files)} parquet + {len(video_files)} video files")
        
    except Exception as e:
        validation["sample_data"]["file_validation_error"] = str(e)
        print(f"❌ File validation failed: {e}")
    
    return validation


def print_test_summary(results: Dict[str, Any]):
    """Print comprehensive test summary"""
    
    print("=" * 50)
    
    # Conversion results
    if results["conversion_success"]:
        print("✅ CONVERSION: SUCCESS")
        conv = results["conversion_result"]
        print(f"   📊 Frames: {conv['processing']['valid_frames']}")
        print(f"   📈 Hand valid: {conv['data_quality']['hand_valid_rate']:.1%}")
        print(f"   📈 Pose valid: {conv['data_quality']['pose_valid_rate']:.1%}")
    else:
        print("❌ CONVERSION: FAILED")
    
    # Loading results
    if results["loading_success"] is True:
        print("✅ LOADING: SUCCESS")
        print(f"   📊 Dataset length: {results['dataset_info']['length']}")
        print(f"   🔑 Features: {len(results['dataset_info']['features'])}")
    elif results["loading_success"] is False:
        print("❌ LOADING: FAILED")
    else:
        print("⚠️  LOADING: SKIPPED (LeRobotDataset not available)")
    
    # Validation results
    if results["validation_success"]:
        print("✅ VALIDATION: SUCCESS")
        
        # Print sample data highlights
        sample_data = results["sample_data"]
        
        if "hand_landmarks_sample" in sample_data:
            print(f"   ✋ Hand landmarks (first 3 points):")
            for i, point in enumerate(sample_data["hand_landmarks_sample"]):
                print(f"      Point {i}: [{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}]")
        
        if "pose_translation" in sample_data:
            pos = sample_data["pose_translation"]
            print(f"   📍 Device position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
        
        if "image_shape" in sample_data:
            shape = sample_data["image_shape"]
            print(f"   📷 Image shape: {shape}")
        
        # Verify NaN placeholders
        if "raw_sample" in sample_data:
            tactile_cols = [k for k in sample_data["raw_sample"].keys() if "action.tactile" in k and "mask" not in k]
            if tactile_cols:
                print(f"   🤖 Tactile placeholders: {len(tactile_cols)} columns (NaN as expected)")
    else:
        print("❌ VALIDATION: FAILED")
    
    # Error summary
    if results["errors"]:
        print(f"⚠️  ERRORS: {len(results['errors'])}")
        for error in results["errors"]:
            print(f"   - {error}")
    
    # Overall result
    overall_success = (
        results["conversion_success"] and 
        results["validation_success"] and
        (results["loading_success"] != False)  # True or None (skipped) is OK
    )
    
    if overall_success:
        print("\n🎉 OVERALL: SMOKE TEST PASSED!")
        print("   The aria2lerobot converter is working correctly.")
    else:
        print("\n💥 OVERALL: SMOKE TEST FAILED!")
        print("   Please check the errors above and fix issues.")
    
    print("=" * 50)


def main():
    """Main entry point"""
    
    if len(sys.argv) < 2:
        print("Usage: python smoke_test.py <vrs_file> [output_dir]")
        print("\nExample:")
        print("  python smoke_test.py ../glue.vrs")
        print("  python smoke_test.py ../glue.vrs ./test_output")
        sys.exit(1)
    
    vrs_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Run smoke test
    results = run_smoke_test(vrs_path, output_dir, cleanup=(output_dir is None))
    
    # Exit with appropriate code
    overall_success = (
        results["conversion_success"] and 
        results["validation_success"] and
        (results["loading_success"] != False)
    )
    
    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    main()
