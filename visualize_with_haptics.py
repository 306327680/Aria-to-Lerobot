#!/usr/bin/env python3
"""
Aria数据可视化工具（支持触觉数据）

生成带有手部关键点、眼动数据投影和触觉力度指示条的可视化视频
"""

import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import argparse
from tqdm import tqdm

# Add aria_api to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aria_api import AriaSession


class HapticVisualizer:
    """触觉数据可视化器"""

    def __init__(self, csv_path: str, time_offset: float = 0.0, max_force: float = 150.0):
        """
        Args:
            csv_path: 触觉CSV文件路径
            time_offset: 时间偏移（秒）
            max_force: 最大力度值（用于归一化显示）
        """
        self.csv_path = csv_path
        self.time_offset = time_offset
        self.max_force = max_force

        # 加载CSV
        df = pd.read_csv(csv_path)

        # 提取时间戳
        if 'relative_time' in df.columns:
            self.timestamps = df['relative_time'].values
        elif 'timestamp' in df.columns:
            timestamps = df['timestamp'].values
            if timestamps[0] > 1e9:
                timestamps = timestamps / 1e9
            self.timestamps = timestamps - timestamps[0]
        else:
            raise ValueError("CSV must have 'relative_time' or 'timestamp' column")

        # 提取力度数据
        magnitude_cols = [col for col in df.columns if 'magnitude' in col.lower()]

        if len(magnitude_cols) >= 2:
            self.force1 = df[magnitude_cols[0]].values
            self.force2 = df[magnitude_cols[1]].values
            self.sensor1_name = magnitude_cols[0].replace('_magnitude', '').replace('_', ' ')
            self.sensor2_name = magnitude_cols[1].replace('_magnitude', '').replace('_', ' ')
        elif len(magnitude_cols) == 1:
            self.force1 = df[magnitude_cols[0]].values
            self.force2 = np.zeros_like(self.force1)
            self.sensor1_name = magnitude_cols[0].replace('_magnitude', '').replace('_', ' ')
            self.sensor2_name = "Sensor 2"
        else:
            raise ValueError(f"Cannot find force columns in {csv_path}")

        print(f"触觉可视化器初始化:")
        print(f"  传感器1: {self.sensor1_name}")
        print(f"  传感器2: {self.sensor2_name}")
        print(f"  最大力度: {max_force}")
        print(f"  时间偏移: {time_offset:.3f}s")

    def get_force_at_time(self, time_s: float) -> Tuple[float, float]:
        """获取指定时间的力度值（线性插值）"""
        query_time = time_s - self.time_offset

        if query_time < self.timestamps[0]:
            return 0.0, 0.0
        elif query_time > self.timestamps[-1]:
            return float(self.force1[-1]), float(self.force2[-1])

        idx = np.searchsorted(self.timestamps, query_time)
        if idx == 0:
            return float(self.force1[0]), float(self.force2[0])

        t0, t1 = self.timestamps[idx-1], self.timestamps[idx]
        f1_0, f1_1 = self.force1[idx-1], self.force1[idx]
        f2_0, f2_1 = self.force2[idx-1], self.force2[idx]

        alpha = (query_time - t0) / (t1 - t0)
        force1 = f1_0 + alpha * (f1_1 - f1_0)
        force2 = f2_0 + alpha * (f2_1 - f2_0)

        return float(force1), float(force2)

    def draw_force_bars(self, image: np.ndarray, force1: float, force2: float) -> np.ndarray:
        """
        在图像上绘制触觉力度指示条

        Args:
            image: 输入图像
            force1: 传感器1的力度值
            force2: 传感器2的力度值

        Returns:
            带有力度指示条的图像
        """
        h, w = image.shape[:2]

        # 指示条参数
        bar_width = 30
        bar_height = 200
        bar_margin = 20
        bar_x1 = w - bar_margin - 2 * bar_width - 15  # 第一个条的X位置
        bar_x2 = w - bar_margin - bar_width            # 第二个条的X位置
        bar_y = bar_margin

        # 归一化力度值
        force1_norm = np.clip(force1 / self.max_force, 0, 1)
        force2_norm = np.clip(force2 / self.max_force, 0, 1)

        # 绘制第一个力度条
        self._draw_single_bar(
            image, bar_x1, bar_y, bar_width, bar_height,
            force1_norm, force1, self.sensor1_name, (0, 165, 255)  # 橙色
        )

        # 绘制第二个力度条
        self._draw_single_bar(
            image, bar_x2, bar_y, bar_width, bar_height,
            force2_norm, force2, self.sensor2_name, (0, 255, 0)  # 绿色
        )

        return image

    def _draw_single_bar(self, image: np.ndarray, x: int, y: int,
                        width: int, height: int, value_norm: float,
                        value_raw: float, label: str, color: Tuple[int, int, int]):
        """绘制单个力度指示条"""

        # 背景框
        cv2.rectangle(image, (x, y), (x + width, y + height), (50, 50, 50), -1)
        cv2.rectangle(image, (x, y), (x + width, y + height), (200, 200, 200), 2)

        # 填充力度条
        fill_height = int(height * value_norm)
        if fill_height > 0:
            fill_y = y + height - fill_height
            cv2.rectangle(image, (x, fill_y), (x + width, y + height), color, -1)

        # 绘制刻度线
        num_ticks = 5
        for i in range(num_ticks + 1):
            tick_y = y + int(height * i / num_ticks)
            cv2.line(image, (x - 5, tick_y), (x, tick_y), (200, 200, 200), 1)

            # 刻度标签
            tick_value = self.max_force * (1 - i / num_ticks)
            if i % 2 == 0:  # 只显示偶数刻度
                label_text = f"{int(tick_value)}"
                (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)
                cv2.putText(image, label_text,
                          (x - text_w - 8, tick_y + text_h // 2),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 200), 1)

        # 标签
        label_parts = label.split()
        for i, part in enumerate(label_parts):
            cv2.putText(image, part,
                      (x + 2, y + height + 15 + i * 12),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

        # 当前数值
        value_text = f"{value_raw:.1f}"
        (text_w, text_h), _ = cv2.getTextSize(value_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        text_x = x + (width - text_w) // 2
        text_y = y - 8
        cv2.putText(image, value_text, (text_x, text_y),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def create_visualization_with_haptics(
    vrs_path: str,
    output_path: str,
    haptic_csv: Optional[str] = None,
    haptic_time_offset: float = 0.0,
    max_force: float = 150.0,
    max_frames: Optional[int] = None,
    stride: int = 1
) -> None:
    """
    创建带触觉数据的可视化视频

    Args:
        vrs_path: VRS文件路径
        output_path: 输出视频路径
        haptic_csv: 触觉CSV文件路径
        haptic_time_offset: 触觉时间偏移（秒）
        max_force: 最大力度值
        max_frames: 最大帧数限制
        stride: 帧间隔
    """
    print(f"创建可视化视频...")
    print(f"VRS文件: {vrs_path}")
    print(f"输出路径: {output_path}")

    # 初始化Aria session
    session = AriaSession.from_vrs(vrs_path)

    # 初始化触觉可视化器
    haptic_viz = None
    if haptic_csv:
        haptic_viz = HapticVisualizer(haptic_csv, haptic_time_offset, max_force)

    # 获取会话信息
    info = session.get_session_info()
    start_ns, end_ns = info['trajectory_time_range_ns']
    start_time_s = start_ns / 1e9

    # 获取相机流
    camera_label = "camera-rgb"

    # 使用trajectory时间戳采样
    sample_interval_ns = int(1e9 / 10.0)  # 10Hz
    timestamps = []
    current_ns = start_ns
    while current_ns <= end_ns:
        timestamps.append(current_ns)
        current_ns += sample_interval_ns

    if max_frames:
        timestamps = timestamps[:max_frames]

    print(f"总帧数: {len(timestamps)}")

    # 初始化视频写入器
    writer = None
    fps = 10.0

    processed_frames = 0

    # 处理帧
    for frame_idx, timestamp_ns in enumerate(tqdm(timestamps[::stride], desc="Processing frames")):
        try:
            # 获取图像
            image_result = session.get_image(camera_label, timestamp_ns)
            if not image_result:
                continue

            img_timestamp, image, intrinsics = image_result

            # 转换为BGR用于OpenCV
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 初始化视频写入器
            if writer is None:
                h, w = image_bgr.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                print(f"初始化视频写入器: {w}x{h} @ {fps}fps")

            # 计算相对时间
            time_s = (timestamp_ns - start_ns) / 1e9

            # 绘制触觉指示条
            if haptic_viz:
                force1, force2 = haptic_viz.get_force_at_time(time_s)
                image_bgr = haptic_viz.draw_force_bars(image_bgr, force1, force2)

            # 添加时间戳信息
            time_text = f"Time: {time_s:.2f}s"
            cv2.putText(image_bgr, time_text, (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 写入帧
            writer.write(image_bgr)
            processed_frames += 1

        except Exception as e:
            print(f"⚠️ 帧 {frame_idx} 处理失败: {e}")
            continue

    # 释放资源
    if writer:
        writer.release()

    print(f"✅ 视频创建完成!")
    print(f"   处理帧数: {processed_frames}")
    print(f"   输出文件: {output_path}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='创建Aria可视化视频（含触觉数据）')
    parser.add_argument('vrs_file', help='VRS文件路径')
    parser.add_argument('output_video', help='输出视频路径')
    parser.add_argument('--haptic-csv', help='触觉CSV文件路径')
    parser.add_argument('--haptic-offset', type=float, default=0.0,
                       help='触觉数据时间偏移（秒）')
    parser.add_argument('--max-force', type=float, default=150.0,
                       help='最大力度值')
    parser.add_argument('--max-frames', type=int, help='最大帧数限制')
    parser.add_argument('--stride', type=int, default=1, help='帧间隔')

    args = parser.parse_args()

    # 检查输入文件
    if not Path(args.vrs_file).exists():
        print(f"❌ VRS文件不存在: {args.vrs_file}")
        return

    if args.haptic_csv and not Path(args.haptic_csv).exists():
        print(f"❌ 触觉CSV文件不存在: {args.haptic_csv}")
        return

    try:
        create_visualization_with_haptics(
            vrs_path=args.vrs_file,
            output_path=args.output_video,
            haptic_csv=args.haptic_csv,
            haptic_time_offset=args.haptic_offset,
            max_force=args.max_force,
            max_frames=args.max_frames,
            stride=args.stride
        )

        print(f"\n🎉 可视化完成!")
        print(f"📹 视频文件: {args.output_video}")

    except Exception as e:
        print(f"❌ 可视化失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()