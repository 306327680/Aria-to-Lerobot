#!/usr/bin/env python3
"""
将Project Aria数据转换为LeRobot数据集格式（支持触觉数据）

扩展功能：
1. 支持添加触觉传感器数据（CSV格式）
2. 自动时间对齐（通过时间偏移参数）
3. 在parquet中包含触觉力度信息
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
import cv2

# 添加aria_api到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from aria_api import AriaSession


class HapticDataLoader:
    """触觉数据加载器"""

    def __init__(self, csv_path: str, time_offset: float = 0.0):
        """
        Args:
            csv_path: 触觉CSV文件路径
            time_offset: 时间偏移（秒），触觉数据需要延后的时间
        """
        self.csv_path = csv_path
        self.time_offset = time_offset

        # 加载CSV数据
        df = pd.read_csv(csv_path)

        # 提取时间戳（使用relative_time列）
        if 'relative_time' in df.columns:
            self.timestamps = df['relative_time'].values
        elif 'timestamp' in df.columns:
            timestamps = df['timestamp'].values
            # 转换为相对时间
            if timestamps[0] > 1e9:  # 可能是绝对时间戳
                timestamps = timestamps / 1e9
            self.timestamps = timestamps - timestamps[0]
        else:
            raise ValueError("CSV file must have 'relative_time' or 'timestamp' column")

        # 提取力度数据（查找magnitude列）
        magnitude_cols = [col for col in df.columns if 'magnitude' in col.lower()]

        if len(magnitude_cols) >= 2:
            # 假设前两个magnitude列分别是两个传感器
            self.force1 = df[magnitude_cols[0]].values
            self.force2 = df[magnitude_cols[1]].values
            self.sensor1_name = magnitude_cols[0].replace('_magnitude', '')
            self.sensor2_name = magnitude_cols[1].replace('_magnitude', '')
        elif len(magnitude_cols) == 1:
            self.force1 = df[magnitude_cols[0]].values
            self.force2 = np.zeros_like(self.force1)
            self.sensor1_name = magnitude_cols[0].replace('_magnitude', '')
            self.sensor2_name = "sensor2"
        else:
            raise ValueError(f"Cannot find force magnitude columns in {csv_path}")

        print(f"触觉数据加载完成:")
        print(f"  传感器1: {self.sensor1_name}")
        print(f"  传感器2: {self.sensor2_name}")
        print(f"  样本数: {len(self.timestamps)}")
        print(f"  时间范围: {self.timestamps[0]:.2f}s - {self.timestamps[-1]:.2f}s")
        print(f"  时间偏移: {time_offset:.3f}s")

    def get_force_at_time(self, time_s: float) -> Tuple[float, float]:
        """
        获取指定时间的力度值（线性插值）

        Args:
            time_s: 时间戳（秒）

        Returns:
            (force1, force2): 两个传感器的力度值
        """
        # 应用时间偏移
        query_time = time_s - self.time_offset

        # 边界检查
        if query_time < self.timestamps[0]:
            # 在触觉数据开始之前，返回0
            return 0.0, 0.0
        elif query_time > self.timestamps[-1]:
            # 超出范围，返回最后一个值
            return float(self.force1[-1]), float(self.force2[-1])

        # 线性插值
        idx = np.searchsorted(self.timestamps, query_time)
        if idx == 0:
            return float(self.force1[0]), float(self.force2[0])

        # 插值
        t0, t1 = self.timestamps[idx-1], self.timestamps[idx]
        f1_0, f1_1 = self.force1[idx-1], self.force1[idx]
        f2_0, f2_1 = self.force2[idx-1], self.force2[idx]

        alpha = (query_time - t0) / (t1 - t0)
        force1 = f1_0 + alpha * (f1_1 - f1_0)
        force2 = f2_0 + alpha * (f2_1 - f2_0)

        return float(force1), float(force2)


def create_lerobot_dataset_with_haptics(
    vrs_file: str,
    output_dir: str,
    haptic_csv: Optional[str] = None,
    haptic_time_offset: float = 0.0,
    sample_rate_hz: float = 10.0,
    image_size: tuple = (224, 224),
):
    """
    将Aria数据（含触觉）转换为LeRobot格式

    Args:
        vrs_file: VRS文件路径
        output_dir: 输出目录
        haptic_csv: 触觉CSV文件路径（可选）
        haptic_time_offset: 触觉数据时间偏移（秒）
        sample_rate_hz: 采样率(Hz)
        image_size: 图像尺寸(width, height)
    """

    print(f"🚀 开始转换Aria数据到LeRobot格式（含触觉数据）")
    print(f"📁 VRS文件: {vrs_file}")
    print(f"📂 输出目录: {output_dir}")
    if haptic_csv:
        print(f"🤚 触觉CSV: {haptic_csv}")
        print(f"⏱️ 时间偏移: {haptic_time_offset:.3f}s")

    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 加载触觉数据（如果提供）
    haptic_loader = None
    if haptic_csv and os.path.exists(haptic_csv):
        haptic_loader = HapticDataLoader(haptic_csv, haptic_time_offset)

    # 初始化Aria会话
    print("🔄 初始化Aria会话...")
    session = AriaSession.from_vrs(vrs_file)

    # 获取会话信息
    info = session.get_session_info()
    print(f"📊 数据概览:")
    print(f"   📷 相机: {len(info['capabilities']['available_cameras'])}")
    print(f"   🗺️ 轨迹: {info.get('trajectory_count', 0)} 点")

    # 计算采样时间戳
    start_ns, end_ns = info['trajectory_time_range_ns']
    sample_interval_ns = int(1e9 / sample_rate_hz)

    timestamps = []
    current_ns = start_ns
    while current_ns <= end_ns:
        timestamps.append(current_ns)
        current_ns += sample_interval_ns

    print(f"🎯 采样 {len(timestamps)} 个时间点 @ {sample_rate_hz} Hz")

    # 初始化数据存储
    data_records = []

    # 遍历时间戳收集数据
    print("📊 开始数据采集...")
    valid_samples = 0

    for frame_id, timestamp_ns in enumerate(tqdm(timestamps, desc="采集数据")):
        try:
            # 获取时间（秒）
            time_s = (timestamp_ns - start_ns) / 1e9

            # 获取图像数据
            image_result = session.get_image("camera-rgb", timestamp_ns)
            if not image_result:
                continue

            img_timestamp, image, intrinsics = image_result

            # 调整图像尺寸
            if image.shape[:2] != image_size[::-1]:
                image = cv2.resize(image, image_size)

            # 获取手部数据
            hands_data = session.nearest_hands(timestamp_ns)
            right_thumb_idx_dist = 0.0

            if hands_data and hands_data.right_hand:
                right_hand = hands_data.right_hand
                if right_hand.wrist_transform:
                    # 获取世界坐标
                    landmarks = right_hand.landmarks_device
                    world_points = right_hand.wrist_transform.transform_points(landmarks)

                    # 计算拇指-食指距离
                    thumb_pos = world_points[4]  # 拇指尖
                    index_pos = world_points[8]  # 食指尖
                    right_thumb_idx_dist = float(np.linalg.norm(thumb_pos - index_pos))

            # 获取触觉数据
            haptic_force1 = 0.0
            haptic_force2 = 0.0
            if haptic_loader:
                haptic_force1, haptic_force2 = haptic_loader.get_force_at_time(time_s)

            # 创建记录
            record = {
                'frame_id': frame_id,
                'timestamp_ns': timestamp_ns,
                'time_s': time_s,
                'image_path': f'images/frame_{frame_id:06d}.jpg',
                'right_thumb_index_distance': right_thumb_idx_dist,
                'haptic_force_sensor1': haptic_force1,
                'haptic_force_sensor2': haptic_force2,
            }

            data_records.append(record)

            # 保存图像
            image_path = output_path / record['image_path']
            image_path.parent.mkdir(exist_ok=True, parents=True)
            cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            valid_samples += 1

        except Exception as e:
            print(f"⚠️ 时间戳 {timestamp_ns} 处理失败: {e}")
            continue

    print(f"✅ 成功采集 {valid_samples} 个有效样本")

    # 保存为Parquet
    save_parquet_dataset(data_records, output_path, info, haptic_loader, vrs_file)

    print(f"🎉 转换完成！数据保存到: {output_path}")
    return output_path


def save_parquet_dataset(records: List[Dict], output_path: Path,
                         info: Dict, haptic_loader: Optional[HapticDataLoader],
                         vrs_file: str):
    """保存为Parquet格式"""

    print("💾 保存Parquet数据集...")

    # 转换为DataFrame
    df = pd.DataFrame(records)

    # 定义schema
    schema = pa.schema([
        ('frame_id', pa.int64()),
        ('timestamp_ns', pa.int64()),
        ('time_s', pa.float64()),
        ('image_path', pa.string()),
        ('right_thumb_index_distance', pa.float32()),
        ('haptic_force_sensor1', pa.float32()),
        ('haptic_force_sensor2', pa.float32()),
    ])

    # 转换为Arrow Table
    table = pa.Table.from_pandas(df, schema=schema)

    # 保存Parquet文件
    parquet_path = output_path / "data.parquet"
    pq.write_table(table, parquet_path, compression='snappy')

    print(f"✅ Parquet文件保存: {parquet_path}")
    print(f"   行数: {len(df)}")
    print(f"   列数: {len(df.columns)}")

    # 保存元数据
    metadata = {
        "dataset_name": f"aria_haptic_{Path(vrs_file).stem}",
        "total_frames": len(records),
        "fps": 10.0,
        "columns": list(df.columns),
        "aria_info": {
            "trajectory_count": info.get('trajectory_count', 0),
            "gaze_count": info.get('gaze_stats', {}).get('count', 0),
            "hand_count": info.get('hand_stats', {}).get('count', 0),
        }
    }

    if haptic_loader:
        metadata["haptic_info"] = {
            "sensor1_name": haptic_loader.sensor1_name,
            "sensor2_name": haptic_loader.sensor2_name,
            "time_offset_s": haptic_loader.time_offset,
            "csv_path": haptic_loader.csv_path
        }

    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ 元数据保存: {output_path / 'metadata.json'}")


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description='将Aria数据转换为LeRobot格式（支持触觉数据）')
    parser.add_argument('vrs_file', help='VRS文件路径')
    parser.add_argument('output_dir', help='输出目录')
    parser.add_argument('--haptic-csv', help='触觉CSV文件路径')
    parser.add_argument('--haptic-offset', type=float, default=0.0,
                       help='触觉数据时间偏移（秒）')
    parser.add_argument('--fps', type=float, default=10.0, help='采样率(Hz)')
    parser.add_argument('--size', type=int, nargs=2, default=[224, 224],
                       help='图像尺寸 (width height)')

    args = parser.parse_args()

    # 检查输入文件
    if not os.path.exists(args.vrs_file):
        print(f"❌ VRS文件不存在: {args.vrs_file}")
        return

    if args.haptic_csv and not os.path.exists(args.haptic_csv):
        print(f"❌ 触觉CSV文件不存在: {args.haptic_csv}")
        return

    try:
        # 执行转换
        output_path = create_lerobot_dataset_with_haptics(
            vrs_file=args.vrs_file,
            output_dir=args.output_dir,
            haptic_csv=args.haptic_csv,
            haptic_time_offset=args.haptic_offset,
            sample_rate_hz=args.fps,
            image_size=tuple(args.size),
        )

        print(f"\n🎉 转换成功！")
        print(f"📂 输出目录: {output_path}")
        print(f"\n📊 生成的文件:")
        print(f"   - data.parquet: 主数据文件")
        print(f"   - metadata.json: 元数据信息")
        print(f"   - images/: 图像文件夹")

    except Exception as e:
        print(f"❌ 转换失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()