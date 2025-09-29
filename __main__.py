"""
CLI entry point for aria2lerobot conversion

Provides command-line interface for converting Aria data to LeRobot format.
"""

import sys
import argparse
from pathlib import Path
from typing import Optional, List

from .convert import convert


def parse_args():
    """Parse command line arguments"""
    
    parser = argparse.ArgumentParser(
        description="Convert Project Aria VRS/MPS data to LeRobot dataset format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic conversion (auto-detect MPS directory)
  python -m aria2lerobot glue.vrs output_dataset/
  
  # Specify MPS directory explicitly  
  python -m aria2lerobot glue.vrs output_dataset/ --mps-dir mps_glue_vrs/
  
  # Use legacy v2.1 format
  python -m aria2lerobot glue.vrs output_dataset/ --legacy-v21
  
  # Custom sampling and quality settings
  python -m aria2lerobot glue.vrs output_dataset/ --fps 15 --stride 2 --max-frames 10000
        """
    )
    
    # Required arguments
    parser.add_argument("vrs_path", type=str, help="Path to VRS file")
    parser.add_argument("output_dir", type=str, help="Output directory for LeRobot dataset")
    
    # Optional arguments
    parser.add_argument("--mps-dir", type=str, help="MPS directory path (auto-detect if not specified)")
    parser.add_argument("--fps", type=float, help="Target FPS (auto-detect if not specified)")
    parser.add_argument("--max-slop-ns", type=int, default=5_000_000, 
                       help="Maximum time alignment tolerance in nanoseconds (default: 5ms)")
    
    # Format options
    parser.add_argument("--legacy-v21", action="store_true", 
                       help="Use LeRobot v2.1 format instead of v3.0")
    
    # Processing options
    parser.add_argument("--stride", type=int, default=1,
                       help="Frame stride (1=every frame, 2=every other frame, etc.)")
    parser.add_argument("--max-frames", type=int,
                       help="Maximum number of frames to process")
    parser.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing output directory")
    
    # Episode splitting (future feature)
    parser.add_argument("--episode-splits", type=int, nargs="*",
                       help="Frame indices to split episodes (space-separated)")
    
    # Logging
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Logging level")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    
    return parser.parse_args()


def auto_detect_mps_dir(vrs_path: Path) -> Optional[Path]:
    """Auto-detect MPS directory based on VRS filename"""
    vrs_stem = vrs_path.stem
    vrs_parent = vrs_path.parent
    
    # Common MPS directory patterns
    patterns = [
        f"mps_{vrs_stem}_vrs",
        f"mps_{vrs_stem}",
        f"{vrs_stem}_mps",
        f"mps_{vrs_stem}_output",
    ]
    
    for pattern in patterns:
        candidate = vrs_parent / pattern
        if candidate.exists() and candidate.is_dir():
            # Check if it looks like an MPS directory
            if any((candidate / subdir).exists() for subdir in ["slam", "hand_tracking", "eye_gaze"]):
                return candidate
    
    return None


def validate_inputs(args) -> List[str]:
    """Validate input arguments"""
    errors = []
    
    # Check VRS file
    vrs_path = Path(args.vrs_path)
    if not vrs_path.exists():
        errors.append(f"VRS file not found: {args.vrs_path}")
    elif not vrs_path.is_file():
        errors.append(f"VRS path is not a file: {args.vrs_path}")
    elif vrs_path.suffix.lower() != '.vrs':
        errors.append(f"File should have .vrs extension: {args.vrs_path}")
    
    # Check/detect MPS directory
    if args.mps_dir:
        mps_dir = Path(args.mps_dir)
        if not mps_dir.exists():
            errors.append(f"MPS directory not found: {args.mps_dir}")
        elif not mps_dir.is_dir():
            errors.append(f"MPS path is not a directory: {args.mps_dir}")
    else:
        # Auto-detect
        detected_mps = auto_detect_mps_dir(vrs_path)
        if detected_mps is None:
            errors.append(f"Cannot auto-detect MPS directory for {args.vrs_path}. Please specify --mps-dir.")
        else:
            args.mps_dir = str(detected_mps)
            print(f"Auto-detected MPS directory: {detected_mps}")
    
    # Check output directory
    out_dir = Path(args.output_dir)
    if out_dir.exists() and not args.overwrite:
        errors.append(f"Output directory exists. Use --overwrite to replace: {args.output_dir}")
    
    # Validate numeric parameters
    if args.fps is not None and args.fps <= 0:
        errors.append("FPS must be positive")
    
    if args.stride <= 0:
        errors.append("Stride must be positive")
    
    if args.max_frames is not None and args.max_frames <= 0:
        errors.append("Max frames must be positive")
    
    if args.max_slop_ns < 0:
        errors.append("Max slop must be non-negative")
    
    return errors


def main():
    """Main CLI entry point"""
    
    args = parse_args()
    
    # Validate inputs
    errors = validate_inputs(args)
    if errors:
        print("❌ Validation errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    
    # Print configuration
    if not args.quiet:
        print("🚀 Aria to LeRobot Conversion")
        print(f"📁 VRS file: {args.vrs_path}")
        print(f"📂 MPS directory: {args.mps_dir}")
        print(f"📤 Output: {args.output_dir}")
        print(f"🎯 Format: LeRobot v{'2.1' if args.legacy_v21 else '3.0'}")
        if args.fps:
            print(f"🎬 FPS: {args.fps}")
        if args.stride > 1:
            print(f"⏭️  Stride: {args.stride}")
        if args.max_frames:
            print(f"🔢 Max frames: {args.max_frames:,}")
        print()
    
    try:
        # Execute conversion
        result = convert(
            vrs_path=args.vrs_path,
            mps_dir=args.mps_dir,
            out_dir=args.output_dir,
            max_slop_ns=args.max_slop_ns,
            fps=args.fps,
            episode_splits=args.episode_splits,
            legacy_v21=args.legacy_v21,
            overwrite=args.overwrite,
            stride=args.stride,
            max_frames=args.max_frames,
            log_level=args.log_level,
        )
        
        if not args.quiet:
            print("\n✅ Conversion completed successfully!")
            print(f"📊 Processed: {result['processing']['valid_frames']:,} frames")
            print(f"⏱️  Time: {result['processing']['elapsed_time_s']:.1f}s")
            print(f"🗂️  Files: {result['files']['parquet_shards']} parquet + {result['files']['video_shards']} video")
            print(f"📈 Hand valid rate: {result['data_quality']['hand_valid_rate']:.1%}")
            print(f"📈 Pose valid rate: {result['data_quality']['pose_valid_rate']:.1%}")
            print(f"📂 Output directory: {args.output_dir}")
        
        sys.exit(0)
        
    except KeyboardInterrupt:
        print("\n⚠️ Conversion interrupted by user")
        sys.exit(130)
        
    except Exception as e:
        print(f"\n❌ Conversion failed: {e}")
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
