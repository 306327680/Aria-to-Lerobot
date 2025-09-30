"""
Parquet I/O utilities for incremental writing and sharding

Handles incremental parquet writing, sharding, and statistics computation
for LeRobot dataset conversion.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterator, Tuple
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from dataclasses import dataclass
import json

from .schema import LeRobotSchema, validate_schema_data
from .utils import compute_statistics


logger = logging.getLogger(__name__)


@dataclass
class ShardConfig:
    """Configuration for parquet sharding"""
    max_rows_per_shard: int = 50000
    max_size_mb: int = 100
    compression: str = "zstd"
    compression_level: Optional[int] = None


class ParquetWriter:
    """Incremental parquet writer with sharding support"""
    
    def __init__(self,
                 output_dir: Path,
                 schema: LeRobotSchema,
                 shard_config: Optional[ShardConfig] = None,
                 base_filename: str = "data"):
        """
        Initialize parquet writer
        
        Args:
            output_dir: Output directory for parquet files
            schema: LeRobot schema definition
            shard_config: Sharding configuration
            base_filename: Base filename for shards (e.g., "data" -> "data_000.parquet")
        """
        self.output_dir = Path(output_dir)
        self.schema = schema
        self.shard_config = shard_config or ShardConfig()
        self.base_filename = base_filename
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Current shard state
        self.current_shard = 0
        self.current_batch: List[Dict[str, Any]] = []
        self.total_rows = 0
        self.shard_metadata: List[Dict[str, Any]] = []
        
        # Statistics tracking
        self.column_data: Dict[str, List] = {col: [] for col in schema.columns.keys()}
        
        # PyArrow schema
        self.pa_schema = schema.get_parquet_schema()
        
        logger.info(f"Initialized ParquetWriter: {output_dir} with {len(schema.columns)} columns")
    
    def add_row(self, row_data: Dict[str, Any]):
        """Add a row to the current batch"""
        
        # Validate against schema
        errors = validate_schema_data(row_data, self.schema)
        if errors:
            raise ValueError(f"Schema validation failed: {errors}")
        
        # Convert arrays to lists for JSON serialization
        processed_row = {}
        for col_name, value in row_data.items():
            if col_name in self.schema.video_columns:
                continue  # Skip video columns
            
            if isinstance(value, np.ndarray):
                processed_row[col_name] = value.flatten().tolist()
                # Track for statistics
                if col_name in self.column_data:
                    self.column_data[col_name].extend(value.flatten())
            else:
                processed_row[col_name] = value
                # Track for statistics
                if col_name in self.column_data:
                    self.column_data[col_name].append(value)
        
        self.current_batch.append(processed_row)
        self.total_rows += 1
        
        # Check if we need to flush the shard
        if len(self.current_batch) >= self.shard_config.max_rows_per_shard:
            self._flush_shard()
    
    def _flush_shard(self):
        """Flush current batch to a parquet shard"""
        if not self.current_batch:
            return

        # For single shard (episode), use base_filename without suffix for LeRobot compatibility
        if self.current_shard == 0:
            shard_filename = f"{self.base_filename}.parquet"
        else:
            shard_filename = f"{self.base_filename}_{self.current_shard:03d}.parquet"
        shard_path = self.output_dir / shard_filename
        
        logger.info(f"Writing shard {self.current_shard}: {len(self.current_batch)} rows -> {shard_filename}")
        
        try:
            # Convert batch to PyArrow table
            table = self._batch_to_table(self.current_batch)
            
            # Write parquet file
            pq.write_table(
                table,
                shard_path,
                compression=self.shard_config.compression,
                compression_level=self.shard_config.compression_level,
                write_statistics=True
            )
            
            # Record shard metadata
            file_size = shard_path.stat().st_size
            shard_info = {
                "shard_id": self.current_shard,
                "filename": shard_filename,
                "row_count": len(self.current_batch),
                "file_size_bytes": file_size,
                "columns": list(self.current_batch[0].keys()) if self.current_batch else [],
            }
            self.shard_metadata.append(shard_info)
            
            logger.debug(f"Shard {self.current_shard} written: {file_size} bytes")
            
        except Exception as e:
            logger.error(f"Failed to write shard {self.current_shard}: {e}")
            raise
        
        # Reset batch
        self.current_batch = []
        self.current_shard += 1
    
    def _batch_to_table(self, batch: List[Dict[str, Any]]) -> pa.Table:
        """Convert batch to PyArrow table"""
        if not batch:
            return pa.Table.from_arrays([], schema=self.pa_schema)
        
        # Group data by column
        column_arrays = {}
        for col_name in self.pa_schema.names:
            column_data = [row.get(col_name) for row in batch]
            
            # Handle different data types
            col_spec = self.schema.columns[col_name]
            if len(col_spec.shape) > 0:
                # Array column - already flattened to lists
                column_arrays[col_name] = pa.array(column_data)
            else:
                # Scalar column
                if col_spec.dtype == "bool":
                    column_arrays[col_name] = pa.array(column_data, type=pa.bool_())
                elif col_spec.dtype == "int32":
                    column_arrays[col_name] = pa.array(column_data, type=pa.int32())
                elif col_spec.dtype == "int64":
                    column_arrays[col_name] = pa.array(column_data, type=pa.int64())
                elif col_spec.dtype == "float32":
                    column_arrays[col_name] = pa.array(column_data, type=pa.float32())
                elif col_spec.dtype == "float64":
                    column_arrays[col_name] = pa.array(column_data, type=pa.float64())
                else:
                    column_arrays[col_name] = pa.array(column_data)
        
        return pa.table(column_arrays, schema=self.pa_schema)
    
    def finalize(self) -> Dict[str, Any]:
        """Finalize writing and return metadata"""
        # Flush remaining batch
        if self.current_batch:
            self._flush_shard()
        
        # Compute statistics
        stats = self._compute_statistics()
        
        # Overall metadata
        metadata = {
            "total_rows": self.total_rows,
            "shard_count": len(self.shard_metadata),
            "shards": self.shard_metadata,
            "statistics": stats,
            "schema_version": self.schema.version,
            "output_directory": str(self.output_dir),
        }
        
        logger.info(f"ParquetWriter finalized: {self.total_rows} rows in {len(self.shard_metadata)} shards")
        return metadata
    
    def _compute_statistics(self) -> Dict[str, Any]:
        """Compute statistics for all columns"""
        stats = {}
        
        for col_name, data_list in self.column_data.items():
            if not data_list:
                continue
            
            col_spec = self.schema.columns[col_name]
            
            if col_spec.dtype in ["float32", "float64"]:
                # Continuous data
                data_array = np.array(data_list, dtype=col_spec.dtype)
                stats[col_name] = compute_statistics(data_array, col_name)
            elif col_spec.dtype == "bool":
                # Boolean data
                true_count = sum(data_list)
                stats[col_name] = {
                    "name": col_name,
                    "count": len(data_list),
                    "true_count": true_count,
                    "false_count": len(data_list) - true_count,
                    "true_ratio": true_count / len(data_list) if data_list else 0,
                }
            elif col_spec.dtype in ["int32", "int64"]:
                # Integer data
                data_array = np.array(data_list, dtype=col_spec.dtype)
                stats[col_name] = compute_statistics(data_array.astype(float), col_name)
        
        return stats


def write_episode_metadata(episodes_dir: Path, 
                          episode_data: List[Dict[str, Any]]) -> Path:
    """
    Write episode metadata to parquet files
    
    Args:
        episodes_dir: Directory for episode metadata
        episode_data: List of episode dictionaries
    
    Returns:
        Path to written parquet file
    """
    episodes_dir.mkdir(parents=True, exist_ok=True)
    
    if not episode_data:
        # Create empty file
        empty_table = pa.table({
            "episode_id": pa.array([], type=pa.int32()),
            "start_frame": pa.array([], type=pa.int64()),
            "end_frame": pa.array([], type=pa.int64()),
            "length": pa.array([], type=pa.int64()),
            "task": pa.array([], type=pa.string()),
        })
        output_path = episodes_dir / "episodes.parquet"
        pq.write_table(empty_table, output_path)
        return output_path
    
    # Convert to PyArrow table
    table_data = {
        "episode_id": [ep["episode_id"] for ep in episode_data],
        "start_frame": [ep["start_frame"] for ep in episode_data],
        "end_frame": [ep["end_frame"] for ep in episode_data],
        "length": [ep["length"] for ep in episode_data],
        "task": [ep.get("task", "unknown") for ep in episode_data],
    }
    
    table = pa.table({
        "episode_id": pa.array(table_data["episode_id"], type=pa.int32()),
        "start_frame": pa.array(table_data["start_frame"], type=pa.int64()),
        "end_frame": pa.array(table_data["end_frame"], type=pa.int64()),
        "length": pa.array(table_data["length"], type=pa.int64()),
        "task": pa.array(table_data["task"], type=pa.string()),
    })
    
    output_path = episodes_dir / "episodes.parquet"
    pq.write_table(table, output_path, compression="zstd")
    
    logger.info(f"Written episode metadata: {len(episode_data)} episodes -> {output_path}")
    return output_path


def read_parquet_metadata(parquet_path: Path) -> Dict[str, Any]:
    """Read metadata from parquet file"""
    try:
        parquet_file = pq.ParquetFile(parquet_path)
        metadata = parquet_file.metadata
        
        return {
            "num_rows": metadata.num_rows,
            "num_columns": metadata.num_columns,
            "serialized_size": metadata.serialized_size,
            "schema": parquet_file.schema_arrow,
            "row_groups": metadata.num_row_groups,
        }
    except Exception as e:
        logger.error(f"Failed to read parquet metadata from {parquet_path}: {e}")
        return {"error": str(e)}


def merge_parquet_shards(shard_paths: List[Path], 
                        output_path: Path,
                        delete_shards: bool = False) -> Dict[str, Any]:
    """
    Merge multiple parquet shards into a single file
    
    Args:
        shard_paths: List of shard file paths
        output_path: Output merged file path
        delete_shards: Whether to delete original shards
    
    Returns:
        Merge metadata
    """
    if not shard_paths:
        raise ValueError("No shard paths provided")
    
    logger.info(f"Merging {len(shard_paths)} parquet shards -> {output_path}")
    
    try:
        # Read all shards
        tables = []
        total_rows = 0
        
        for shard_path in shard_paths:
            table = pq.read_table(shard_path)
            tables.append(table)
            total_rows += len(table)
        
        # Concatenate tables
        merged_table = pa.concat_tables(tables)
        
        # Write merged file
        pq.write_table(merged_table, output_path, compression="zstd")
        
        # Get file size
        file_size = output_path.stat().st_size
        
        # Delete shards if requested
        if delete_shards:
            for shard_path in shard_paths:
                shard_path.unlink()
        
        metadata = {
            "output_path": str(output_path),
            "input_shards": len(shard_paths),
            "total_rows": total_rows,
            "file_size_bytes": file_size,
            "shards_deleted": delete_shards,
        }
        
        logger.info(f"Merged {len(shard_paths)} shards: {total_rows} rows, {file_size} bytes")
        return metadata
        
    except Exception as e:
        logger.error(f"Failed to merge parquet shards: {e}")
        raise


def validate_parquet_files(parquet_paths: List[Path],
                          expected_schema: Optional[pa.Schema] = None) -> Dict[str, Any]:
    """
    Validate parquet files for consistency
    
    Args:
        parquet_paths: List of parquet file paths
        expected_schema: Expected PyArrow schema (optional)
    
    Returns:
        Validation results
    """
    results = {
        "valid": True,
        "files": [],
        "total_rows": 0,
        "errors": [],
    }
    
    for path in parquet_paths:
        file_result = {
            "path": str(path),
            "valid": True,
            "errors": [],
        }
        
        try:
            if not path.exists():
                file_result["valid"] = False
                file_result["errors"].append("File does not exist")
                continue
            
            # Read metadata
            parquet_file = pq.ParquetFile(path)
            file_result["num_rows"] = parquet_file.metadata.num_rows
            file_result["num_columns"] = parquet_file.metadata.num_columns
            file_result["schema"] = parquet_file.schema_arrow
            
            results["total_rows"] += file_result["num_rows"]
            
            # Check schema compatibility
            if expected_schema is not None:
                if not file_result["schema"].equals(expected_schema):
                    file_result["valid"] = False
                    file_result["errors"].append("Schema mismatch")
            
        except Exception as e:
            file_result["valid"] = False
            file_result["errors"].append(str(e))
        
        if not file_result["valid"]:
            results["valid"] = False
            results["errors"].extend([f"{path}: {err}" for err in file_result["errors"]])
        
        results["files"].append(file_result)
    
    return results


class ParquetIterator:
    """Iterator for reading parquet data in batches"""
    
    def __init__(self, parquet_paths: List[Path], batch_size: int = 1000):
        """
        Initialize parquet iterator
        
        Args:
            parquet_paths: List of parquet file paths
            batch_size: Number of rows per batch
        """
        self.parquet_paths = parquet_paths
        self.batch_size = batch_size
        self.current_file_idx = 0
        self.current_table = None
        self.current_batch_idx = 0
    
    def __iter__(self):
        return self
    
    def __next__(self) -> pa.Table:
        """Get next batch"""
        # Load next file if needed
        while self.current_table is None or self.current_batch_idx * self.batch_size >= len(self.current_table):
            if self.current_file_idx >= len(self.parquet_paths):
                raise StopIteration
            
            # Load next file
            file_path = self.parquet_paths[self.current_file_idx]
            self.current_table = pq.read_table(file_path)
            self.current_file_idx += 1
            self.current_batch_idx = 0
        
        # Extract batch
        start_idx = self.current_batch_idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.current_table))
        
        batch = self.current_table.slice(start_idx, end_idx - start_idx)
        self.current_batch_idx += 1
        
        return batch
