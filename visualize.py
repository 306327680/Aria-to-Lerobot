#!/usr/bin/env python3
"""
Aria数据可视化工具

生成带有手部关键点和眼动数据投影的可视化视频
"""

import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import argparse
import logging
from tqdm import tqdm

# Add aria_api to path
sys.path.insert(0, str(Path(__file__).parent.parent / "aria_api"))

try:
    from aria_api import AriaSession
except ImportError:
    # If running as standalone script, add current directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from aria_api import AriaSession

try:
    from aria2lerobot.utils import setup_logging
except ImportError:
    # If running as standalone, define a simple setup_logging
    def setup_logging(verbose=False):
        logging.basicConfig(
            level=logging.DEBUG if verbose else logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

logger = logging.getLogger(__name__)


# 手部关键点连接关系 (Project Aria 21点模型)
# 参考: code/hand_tracking_parser/21-keypoints.png
# 关键点布局: 0-4指尖, 5手腕, 6-7拇指关节, 8-10食指关节, 11-13中指关节, 14-16无名指关节, 17-19小指关节, 20手掌中心

# 按手指分组的连接关系，每个手指使用不同颜色
HAND_CONNECTIONS_BY_FINGER = {
    'wrist_palm': {
        'connections': [(5, 20)],  # 手腕到手掌中心
        'color_left': (255, 255, 255),   # 白色
        'color_right': (200, 200, 200)   # 浅灰色
    },
    'thumb': {
        'connections': [(5, 6), (6, 7), (7, 0)],  # 拇指
        'color_left': (255, 0, 0),      # 红色
        'color_right': (200, 0, 0)      # 深红色
    },
    'index': {
        'connections': [(5, 8), (8, 9), (9, 10), (10, 1)],  # 食指
        'color_left': (255, 165, 0),    # 橙色
        'color_right': (200, 120, 0)    # 深橙色
    },
    'middle': {
        'connections': [(5, 11), (11, 12), (12, 13), (13, 2)],  # 中指
        'color_left': (255, 255, 0),    # 黄色
        'color_right': (200, 200, 0)    # 深黄色
    },
    'ring': {
        'connections': [(5, 14), (14, 15), (15, 16), (16, 3)],  # 无名指
        'color_left': (0, 255, 0),      # 绿色
        'color_right': (0, 200, 0)      # 深绿色
    },
    'pinky': {
        'connections': [(5, 17), (17, 18), (18, 19), (19, 4)],  # 小指
        'color_left': (0, 0, 255),      # 蓝色
        'color_right': (0, 0, 200)      # 深蓝色
    },
    'palm_connections': {
        'connections': [(8, 20), (11, 20), (14, 20), (17, 20)],  # 手掌连接
        'color_left': (128, 0, 128),    # 紫色
        'color_right': (100, 0, 100)    # 深紫色
    }
}

def draw_hand_landmarks(image: np.ndarray, 
                       landmarks_2d: np.ndarray, 
                       valid_mask: np.ndarray,
                       is_left_hand: bool = True,
                       thickness: int = 2) -> np.ndarray:
    """
    在图像上绘制手部关键点和连接线（彩虹色）
    
    Args:
        image: 输入图像
        landmarks_2d: 2D关键点坐标 [21, 2]
        valid_mask: 有效性掩码 [21]
        is_left_hand: 是否为左手
        thickness: 线条粗细
    
    Returns:
        带有手部关键点的图像
    """
    image = image.copy()
    
    # 按手指绘制不同颜色的连接线
    for finger_name, finger_info in HAND_CONNECTIONS_BY_FINGER.items():
        # 选择左手或右手的颜色
        color = finger_info['color_left'] if is_left_hand else finger_info['color_right']
        
        # 绘制该手指的所有连接线
        for start_idx, end_idx in finger_info['connections']:
            if valid_mask[start_idx] and valid_mask[end_idx]:
                start_point = tuple(landmarks_2d[start_idx].astype(int))
                end_point = tuple(landmarks_2d[end_idx].astype(int))
                cv2.line(image, start_point, end_point, color, thickness)
    
    # 绘制关键点（白色圆点带黑色边框）
    for i, (x, y) in enumerate(landmarks_2d):
        if valid_mask[i]:
            center = (int(x), int(y))
            # 手腕点用大圆圈，其他关键点用小圆圈
            radius = 6 if i == 5 else 4  # 手腕是索引5
            cv2.circle(image, center, radius, (255, 255, 255), -1)  # 白色填充
            cv2.circle(image, center, radius, (0, 0, 0), 1)         # 黑色边框
            
            # 在手腕处标注
            if i == 5:  # 手腕关键点
                hand_label = "L" if is_left_hand else "R"
                cv2.putText(image, hand_label, (center[0] + 8, center[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cv2.putText(image, hand_label, (center[0] + 8, center[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return image

def draw_gaze_point(image: np.ndarray,
                   gaze_point: Tuple[float, float],
                   color: Tuple[int, int, int] = (0, 0, 255),
                   radius: int = 8) -> np.ndarray:
    """
    在图像上绘制眼动注视点
    
    Args:
        image: 输入图像
        gaze_point: 注视点坐标 (u, v)
        color: 绘制颜色 (B, G, R)
        radius: 圆圈半径
    
    Returns:
        带有眼动点的图像
    """
    image = image.copy()
    center = (int(gaze_point[0]), int(gaze_point[1]))
    
    # 绘制十字准星
    cross_size = radius + 5
    cv2.line(image, 
             (center[0] - cross_size, center[1]), 
             (center[0] + cross_size, center[1]), 
             color, 2)
    cv2.line(image, 
             (center[0], center[1] - cross_size), 
             (center[0], center[1] + cross_size), 
             color, 2)
    
    # 绘制圆圈
    cv2.circle(image, center, radius, color, 2)
    
    # 标注
    cv2.putText(image, "GAZE", (center[0] + 12, center[1] - 12), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return image

def draw_force_bars(image: np.ndarray,
                    force_left: float,
                    force_right: float,
                    max_force: float = 150.0) -> np.ndarray:
    """
    绘制力度指示条

    Args:
        image: 输入图像
        force_left: 左手力度值
        force_right: 右手力度值
        max_force: 最大力度值（用于归一化）

    Returns:
        带有力度指示条的图像
    """
    image = image.copy()
    h, w = image.shape[:2]

    # 力度条参数
    bar_width = 30
    bar_max_height = 150
    bar_x_left = w - 120  # 左手力度条X坐标
    bar_x_right = w - 60  # 右手力度条X坐标
    bar_y_bottom = h - 30  # 底部Y坐标

    # 绘制半透明背景
    overlay = image.copy()
    bg_x1 = w - 140
    bg_y1 = bar_y_bottom - bar_max_height - 40
    bg_x2 = w - 10
    bg_y2 = h - 10
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    # 标题
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, "FORCE", (bg_x1 + 25, bg_y1 + 20),
               font, 0.5, (255, 255, 255), 1)

    # 绘制左手力度条
    force_left_normalized = min(force_left / max_force, 1.0)
    bar_height_left = int(bar_max_height * force_left_normalized)

    # 左手边框
    cv2.rectangle(image,
                 (bar_x_left, bar_y_bottom - bar_max_height),
                 (bar_x_left + bar_width, bar_y_bottom),
                 (100, 100, 100), 2)

    # 左手填充（渐变色：绿->黄->红）
    if bar_height_left > 0:
        # 根据力度值选择颜色
        if force_left_normalized < 0.33:
            color = (0, 255, 0)  # 绿色
        elif force_left_normalized < 0.66:
            color = (0, 255, 255)  # 黄色
        else:
            color = (0, 0, 255)  # 红色

        cv2.rectangle(image,
                     (bar_x_left, bar_y_bottom - bar_height_left),
                     (bar_x_left + bar_width, bar_y_bottom),
                     color, -1)

    # 左手标签和数值
    cv2.putText(image, "L", (bar_x_left + 8, bar_y_bottom + 18),
               font, 0.4, (255, 255, 255), 1)
    cv2.putText(image, f"{force_left:.0f}N", (bar_x_left - 5, bar_y_bottom - bar_max_height - 5),
               font, 0.35, (0, 255, 0), 1)

    # 绘制右手力度条
    force_right_normalized = min(force_right / max_force, 1.0)
    bar_height_right = int(bar_max_height * force_right_normalized)

    # 右手边框
    cv2.rectangle(image,
                 (bar_x_right, bar_y_bottom - bar_max_height),
                 (bar_x_right + bar_width, bar_y_bottom),
                 (100, 100, 100), 2)

    # 右手填充
    if bar_height_right > 0:
        if force_right_normalized < 0.33:
            color = (0, 255, 0)
        elif force_right_normalized < 0.66:
            color = (0, 255, 255)
        else:
            color = (0, 0, 255)

        cv2.rectangle(image,
                     (bar_x_right, bar_y_bottom - bar_height_right),
                     (bar_x_right + bar_width, bar_y_bottom),
                     color, -1)

    # 右手标签和数值
    cv2.putText(image, "R", (bar_x_right + 8, bar_y_bottom + 18),
               font, 0.4, (255, 255, 255), 1)
    cv2.putText(image, f"{force_right:.0f}N", (bar_x_right - 5, bar_y_bottom - bar_max_height - 5),
               font, 0.35, (0, 255, 0), 1)

    return image

def add_info_overlay(image: np.ndarray,
                    frame_idx: int,
                    timestamp_s: float,
                    hands_info: str,
                    gaze_info: str) -> np.ndarray:
    """添加信息叠加层"""
    image = image.copy()
    h, w = image.shape[:2]

    # 半透明背景
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (400, 120), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

    # 文本信息
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    color = (255, 255, 255)
    thickness = 1

    y_offset = 30
    cv2.putText(image, f"Frame: {frame_idx}", (20, y_offset), font, font_scale, color, thickness)
    y_offset += 20
    cv2.putText(image, f"Time: {timestamp_s:.3f}s", (20, y_offset), font, font_scale, color, thickness)
    y_offset += 20
    cv2.putText(image, hands_info, (20, y_offset), font, font_scale, (0, 255, 0), thickness)
    y_offset += 20
    cv2.putText(image, gaze_info, (20, y_offset), font, font_scale, (0, 0, 255), thickness)

    return image

def create_visualization_video(vrs_path: str,
                              mps_dir: str,
                              output_path: str,
                              max_frames: Optional[int] = None,
                              stride: int = 1,
                              rotate_image: bool = False,
                              haptic_loader = None,
                              start_timestamp_ns: int = 0,
                              global_start_timestamp_ns: int = None) -> None:
    """
    创建可视化视频

    Args:
        vrs_path: VRS文件路径
        mps_dir: MPS数据目录
        output_path: 输出视频路径
        max_frames: 最大帧数限制
        stride: 帧间隔
        rotate_image: 是否顺时针旋转90度
        haptic_loader: 触觉数据加载器（可选）
        start_timestamp_ns: 起始时间戳（纳秒），用于定位VRS中的起始帧
        global_start_timestamp_ns: 全局起始时间戳（纳秒），用于计算触觉数据的相对时间。如果为None，则使用start_timestamp_ns
    """
    logger.info(f"创建可视化视频...")
    logger.info(f"VRS文件: {vrs_path}")
    logger.info(f"MPS目录: {mps_dir}")
    logger.info(f"输出路径: {output_path}")

    # If global_start_timestamp_ns is not provided, use start_timestamp_ns
    if global_start_timestamp_ns is None:
        global_start_timestamp_ns = start_timestamp_ns

    # 初始化Aria session
    session = AriaSession.from_paths(vrs_path, mps_dir)
    
    # 获取相机流
    camera_label = "camera-rgb"
    stream_id = session.vrs_data_provider.get_stream_id_from_label(camera_label)
    total_frames = session.vrs_data_provider.get_num_data(stream_id)

    logger.info(f"VRS总帧数: {total_frames}, 步长: {stride}")

    # 找到起始帧索引（基于start_timestamp_ns）
    start_frame_idx = 0
    if start_timestamp_ns > 0:
        # 使用二分搜索快速定位起始帧（避免读取所有帧）
        left, right = 0, total_frames - 1
        found_idx = 0

        while left <= right:
            mid = (left + right) // 2
            try:
                result = session.vrs_data_provider.get_image_data_by_index(stream_id, mid)
                if result is None:
                    # Frame not available, try next
                    left = mid + 1
                    continue

                image_record = result[1]
                mid_timestamp = image_record.capture_timestamp_ns

                if mid_timestamp < start_timestamp_ns:
                    # Target is in the right half
                    left = mid + 1
                elif mid_timestamp >= start_timestamp_ns:
                    # This could be our target, but check if there's an earlier one
                    found_idx = mid
                    right = mid - 1
            except Exception as e:
                logger.debug(f"Error reading frame {mid}: {e}")
                left = mid + 1

        start_frame_idx = found_idx

        # Verify the found frame
        try:
            result = session.vrs_data_provider.get_image_data_by_index(stream_id, start_frame_idx)
            if result is not None:
                image_record = result[1]
                logger.info(f"找到起始帧: idx={start_frame_idx}, ts={image_record.capture_timestamp_ns} (target={start_timestamp_ns})")
        except Exception as e:
            logger.warning(f"无法验证起始帧 {start_frame_idx}: {e}")

    # 计算实际要处理的帧数
    available_frames = total_frames - start_frame_idx
    if max_frames:
        frames_to_process = min(available_frames, max_frames)
    else:
        frames_to_process = available_frames

    logger.info(f"从帧 {start_frame_idx} 开始，处理 {frames_to_process} 帧 (可用帧数: {available_frames})")

    # 初始化视频写入器
    writer = None
    fps = 10.0  # 固定10fps

    processed_frames = 0
    valid_hands = 0
    valid_gaze = 0

    # 处理帧
    for frame_idx in tqdm(range(start_frame_idx, start_frame_idx + frames_to_process, stride), desc="Processing frames"):
        try:
            # 获取图像
            result = session.vrs_data_provider.get_image_data_by_index(stream_id, frame_idx)
            if result is None:
                continue
            
            image_data, image_record = result[0], result[1]
            timestamp_ns = image_record.capture_timestamp_ns
            timestamp_s = timestamp_ns / 1e9
            
            # 转换为numpy数组
            image = image_data.to_numpy_array()
            
            # 确保RGB格式 (VRS提供RGB)
            if len(image.shape) == 3 and image.shape[2] == 3:
                # 转换为BGR用于OpenCV
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
            
            
            # 初始化视频写入器
            if writer is None:
                h, w = image_bgr.shape[:2]
                # 如果需要旋转，调整输出尺寸
                if rotate_image:
                    # 旋转后宽高互换（对于90度旋转）
                    output_w, output_h = h, w
                else:
                    output_w, output_h = w, h
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (output_w, output_h))
                logger.info(f"初始化视频写入器: {output_w}x{output_h} @ {fps}fps")
            
            # 投影手部关键点（始终在原始图像上投影）
            hands_info = "Hands: N/A"
            try:
                hands_result = session.project_hands(timestamp_ns, camera_label)
                if hands_result is not None:
                    left_count = 0
                    right_count = 0
                    
                    # 解析左手数据 (API返回tuple格式)
                    if 'left' in hands_result and hands_result['left'] is not None:
                        left_landmarks, left_valid = hands_result['left']
                        left_count = np.sum(left_valid)
                        if left_count > 0:
                            image_bgr = draw_hand_landmarks(
                                image_bgr, left_landmarks, left_valid, 
                                is_left_hand=True, thickness=2
                            )
                            valid_hands += 1
                    
                    # 解析右手数据 (API返回tuple格式)
                    if 'right' in hands_result and hands_result['right'] is not None:
                        right_landmarks, right_valid = hands_result['right']
                        right_count = np.sum(right_valid)
                        if right_count > 0:
                            image_bgr = draw_hand_landmarks(
                                image_bgr, right_landmarks, right_valid, 
                                is_left_hand=False, thickness=2
                            )
                            valid_hands += 1
                    
                    hands_info = f"Hands: L({left_count}/21) R({right_count}/21)"
            except Exception as e:
                logger.debug(f"手部投影失败: {e}")
            
            # 投影眼动数据（始终在原始图像上投影）
            gaze_info = "Gaze: N/A"
            try:
                gaze_result = session.project_gaze(timestamp_ns, camera_label)
                if gaze_result is not None:
                    # API返回tuple格式: (pixel_coords, depth, valid)
                    gaze_point, depth, is_valid = gaze_result
                    if is_valid:
                        # Ensure gaze_point is properly unpacked and converted to float
                        if isinstance(gaze_point, (tuple, list)):
                            gaze_u, gaze_v = float(gaze_point[0]), float(gaze_point[1])
                        else:
                            # numpy array
                            gaze_u, gaze_v = float(gaze_point[0]), float(gaze_point[1])

                        image_bgr = draw_gaze_point(
                            image_bgr, (gaze_u, gaze_v),
                            color=(0, 0, 255), radius=8
                        )
                        gaze_info = f"Gaze: ({gaze_u:.0f}, {gaze_v:.0f})"
                        valid_gaze += 1
            except Exception as e:
                logger.debug(f"眼动投影失败: {e}")

            # 旋转图像（如果需要）- 在投影后、UI叠加前
            # 这样手部和眼动会随图像旋转，但UI元素保持正常方向
            if rotate_image:
                # 顺时针旋转90度
                image_bgr = cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)

            # 添加信息叠加（在旋转后的图像上）
            image_bgr = add_info_overlay(
                image_bgr, frame_idx, timestamp_s, hands_info, gaze_info
            )

            # 绘制力度指示条（在旋转后的图像上）
            if haptic_loader is not None:
                # 计算相对时间（从全局起始时间戳开始）
                time_s = (timestamp_ns - global_start_timestamp_ns) / 1e9
                # get_force_at_time returns ((mag1, xyz1), (mag2, xyz2))
                (force_left, _), (force_right, _) = haptic_loader.get_force_at_time(time_s)
                image_bgr = draw_force_bars(image_bgr, force_left, force_right)
            
            # 写入视频帧
            writer.write(image_bgr)
            processed_frames += 1
            
        except Exception as e:
            import traceback
            logger.warning(f"处理帧 {frame_idx} 失败: {e}")
            logger.debug(f"详细错误信息:\n{traceback.format_exc()}")
            continue
    
    # 清理
    if writer:
        writer.release()
    
    logger.info(f"可视化视频创建完成!")
    logger.info(f"处理帧数: {processed_frames}")
    logger.info(f"有效手部帧: {valid_hands}")
    logger.info(f"有效眼动帧: {valid_gaze}")
    logger.info(f"输出文件: {output_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="创建Aria数据可视化视频")
    parser.add_argument("vrs_path", help="VRS文件路径")
    parser.add_argument("output_path", help="输出视频路径")
    parser.add_argument("--mps-dir", help="MPS数据目录 (自动检测如果未指定)")
    parser.add_argument("--max-frames", type=int, help="最大帧数限制")
    parser.add_argument("--stride", type=int, default=1, help="帧间隔 (默认: 1)")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    parser.add_argument("--rotate", action="store_true", help="顺时针旋转图像90度")
    
    args = parser.parse_args()
    
    # 设置日志
    log_level = "DEBUG" if args.verbose else "INFO"
    try:
        from aria2lerobot.utils import setup_logging as real_setup_logging
        real_setup_logging(log_level)
    except ImportError:
        setup_logging(args.verbose)
    
    # 验证输入
    vrs_path = Path(args.vrs_path)
    if not vrs_path.exists():
        logger.error(f"VRS文件不存在: {vrs_path}")
        return 1
    
    # 自动检测MPS目录
    if args.mps_dir is None:
        from aria_api.session import AriaSession
        mps_dir = AriaSession._auto_detect_mps_dir(vrs_path)
        if mps_dir is None:
            logger.error(f"无法自动检测MPS目录，请手动指定 --mps-dir")
            return 1
        logger.info(f"自动检测到MPS目录: {mps_dir}")
    else:
        mps_dir = args.mps_dir
    
    # 确保输出目录存在
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        create_visualization_video(
            str(vrs_path),
            str(mps_dir),
            str(output_path),
            max_frames=args.max_frames,
            stride=args.stride,
            rotate_image=args.rotate
        )
        return 0
    except Exception as e:
        logger.error(f"创建可视化视频失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main())
