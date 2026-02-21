# Aria to LeRobot/GR00T Converter

将 [Project Aria](https://projectaria.com/) 眼镜录制的 VRS/MPS 数据（包含 RGB 视频、手部追踪、眼动追踪、设备位姿）与触觉传感器力数据，转换为 [LeRobot v3.0](https://github.com/huggingface/lerobot) 和 [NVIDIA GR00T](https://developer.nvidia.com/isaac-lab) 兼容的数据集格式，用于机器人模仿学习训练。

## 目录

- [前置要求](#前置要求)
- [安装](#安装)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [完整工作流程](#完整工作流程)
- [数据可视化](#数据可视化)
- [数据模态说明](#数据模态说明)
- [参数说明](#参数说明)
- [故障排除](#故障排除)

---

## 前置要求

- Python >= 3.10
- Conda（用于环境管理）
- [Project Aria](https://projectaria.com/) 眼镜采集的 VRS 录制文件
- 已通过 [MPS (Machine Perception Services)](https://facebookresearch.github.io/projectaria_tools/docs/ARK/mps) 处理的输出目录（包含手部追踪、眼动追踪等）
- （可选）触觉传感器 CSV 数据文件

---

## 安装

```bash
# 1. 创建并激活 conda 环境
conda create -n aria2lerobot python=3.10
conda activate aria2lerobot

# 2. 安装 Python 依赖
pip install -r aria2lerobot/requirements.txt

# 3. 安装 Project Aria 工具（用于读取 VRS 文件）
pip install projectaria-tools

# 4. （可选）安装 LeRobot（用于数据集可视化验证）
pip install lerobot

# 5. 设置 PYTHONPATH（确保 aria_api 可访问）
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/aria_api"
```

依赖列表 (`requirements.txt`):

| 包名 | 版本 | 用途 |
|------|------|------|
| numpy | >= 1.21.0 | 数组计算 |
| pyarrow | >= 8.0.0 | Parquet 文件读写 |
| pandas | >= 1.3.0 | CSV/DataFrame 处理 |
| opencv-python | >= 4.5.0 | 视频编码、图像处理 |
| tqdm | >= 4.60.0 | 进度条 |
| orjson | >= 3.8.0 | （可选）快速 JSON 序列化 |
| pytest | >= 6.0.0 | （开发）单元测试 |

---

## 项目结构

```
aria2lerobot/
├── __main__.py                 # CLI 入口
├── convert.py                  # 核心转换逻辑
├── convert_with_haptics.py     # 触觉数据转换
├── schema.py                   # LeRobot schema 定义
├── video.py                    # H.264/H.265 视频编码
├── io_parquet.py               # Parquet 增量写入
├── metadata.py                 # 元数据生成
├── chunking.py                 # Episode 分段
├── frame_processing.py         # 逐帧数据处理
├── utils.py                    # 工具函数
├── visualize.py                # 可视化视频生成
├── visualize_with_haptics.py   # 带触觉数据的可视化
├── scripts/
│   ├── aria_to_groot_pipeline.sh   # 单任务转换流水线
│   └── batch_aria_to_groot.sh      # 批量转换
├── tools/
│   ├── extract_dimensions.py       # 维度提取
│   ├── extract_pinch_distance.py   # 捏合距离提取 (1D)
│   ├── extract_pinch_features.py   # 捏合特征提取 (7D)
│   ├── compute_stats.py            # 统计信息计算
│   ├── load_groot_dataset.py       # 数据集加载测试
│   └── convert_to_groot_format.py  # 格式转换
├── tests/                      # 单元测试
└── examples/                   # 示例代码
```

---

## 快速开始

### 单个任务转换

```bash
bash aria2lerobot/scripts/aria_to_groot_pipeline.sh \
  --vrs glue.vrs \
  --output output_glue \
  --haptic-csv sensor_data/sequence_2_20250927_203243.csv \
  --haptic-offset 1.21 \
  --split-csv aria_haptic_alignment/glue_alignment_segments.csv \
  --task-name "pick and place glue" \
  --test-loading
```

### 批量转换所有任务

```bash
bash aria2lerobot/scripts/batch_aria_to_groot.sh
```

---

## 完整工作流程

### 1. 数据准备

确保你有以下文件：

```
your_data_dir/
├── glue.vrs                    # Aria VRS 录制文件
├── mps_glue_vrs/               # MPS 处理输出
├── sensor_data/
│   └── sequence_2_*.csv        # 触觉传感器数据
└── aria_haptic_alignment/
    └── glue_alignment_segments.csv  # 分段信息
```

### 2. 转换为 LeRobot/GR00T 格式

```bash
bash aria2lerobot/scripts/aria_to_groot_pipeline.sh \
  --vrs glue.vrs \
  --output output_glue \
  --haptic-csv sensor_data/sequence_2_20250927_203243.csv \
  --haptic-offset 1.21 \
  --split-csv aria_haptic_alignment/glue_alignment_segments.csv \
  --task-name "pick and place glue"
```

流水线自动执行：VRS/MPS 转换 → GR00T 元数据生成 → 统计信息计算 → 格式验证。

输出结构：

```
output_glue/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       └── ...
├── videos/
│   └── chunk-000/
│       └── observation.images.main/
│           ├── episode_000000.mp4
│           └── ...
└── meta/
    ├── info.json           # 数据集信息
    ├── tasks.parquet       # 任务列表 (LeRobot)
    ├── tasks.jsonl         # 任务列表 (GR00T)
    ├── episodes/           # 分集信息 (LeRobot)
    ├── episodes.jsonl      # 分集信息 (GR00T)
    ├── stats.json          # 数据统计
    └── modality.json       # 模态映射 (GR00T)
```

### 3. (可选) 创建数据集变体用于实验

从完整数据集提取特定维度，创建不同的实验对照组。

| 数据集名称 | State 维度 | Action 维度 | 用途 |
|-----------|-----------|------------|------|
| `output_glue` | 79 维（手部+姿态） | 8 维（完整力数据） | 完整基线数据集 |
| `output_glue_right_hand_force` | 63 维（右手关键点） | 2 维（力的大小） | 简化的手部+力训练 |
| `output_glue_pinch_distance` | 1 维（拇指-食指距离） | 2 维（力的大小） | 最小化特征实验 |
| `output_glue_pinch_viz` | 7 维（拇指xyz+食指xyz+距离） | 2 维（力的大小） | 捏合动作可视化 |

#### extract_dimensions.py - 维度提取

从完整数据集中提取指定维度的子集。查看完整帮助：`python aria2lerobot/tools/extract_dimensions.py -h`

```bash
# 右手关键点 + 力的大小（GR00T 训练用）
python aria2lerobot/tools/extract_dimensions.py \
  output_glue output_glue_right_hand_force \
  --state-dims 0:63 --action-dims 0,1 \
  --state-name "right_hand" --action-name "force_magnitude" \
  --format groot

# 同样的数据，LeRobot 可视化格式
python aria2lerobot/tools/extract_dimensions.py \
  output_glue output_glue_right_hand_force_viz \
  --state-dims 0:63 --action-dims 0,1 \
  --state-name "right_hand" --action-name "force_magnitude" \
  --format lerobot
```

维度指定格式：单个 `3`、多个 `3,7,10`、范围 `0:63`、混合 `0,5,10:15,20`。

#### extract_pinch_distance.py - 捏合距离

计算拇指尖和食指尖之间的欧几里得距离作为 1D 状态特征。用于测试模型能否仅从捏合距离学习力的控制。

```bash
python aria2lerobot/tools/extract_pinch_distance.py \
  output_glue output_glue_pinch_distance \
  --action-dims 0,1 --format groot
```

#### extract_pinch_features.py - 捏合特征（位置+距离）

提取拇指和食指的 3D 位置加上距离，共 7 维 = thumb_xyz(3) + index_xyz(3) + distance(1)。

```bash
python aria2lerobot/tools/extract_pinch_features.py \
  output_glue output_glue_pinch_viz \
  --action-dims 0,1 --format lerobot
```

### 4. 验证数据集

```bash
# 验证 GR00T 数据加载
python aria2lerobot/tools/load_groot_dataset.py output_glue --sample-index 10

# 可视化检查（需要 LeRobot 格式数据集）
conda activate lerobot
lerobot-dataset-viz \
  --repo-id output_glue_pinch_viz \
  --root /path/to/your/data \
  --episode-index 10
```

### 5. 在 GR00T 中训练

```bash
cd /path/to/Isaac-GR00T
bash train_aria.sh
```

---

## 数据可视化

### 可视化内容

生成的可视化视频包含以下叠加信息：

| 叠加层 | 描述 | 颜色编码 |
|--------|------|----------|
| 手部骨架 | 21 个关键点 + 连接线 | 彩虹色（拇指红、食指橙、中指黄、无名指绿、小指蓝） |
| 手腕标注 | 手腕位置 | 大圆圈 + L/R 标签 |
| 眼动注视点 | 十字准星标注 | 红色十字 + 圆圈 |
| 力度指示条 | 左右手实时力度 | 绿 → 黄 → 红渐变 + 数值 |
| 信息面板 | 帧号、时间戳、统计 | 半透明黑色背景 |

### 使用方法

**在转换流程中生成**：

```bash
python -m aria2lerobot \
  glue.vrs output_glue \
  --haptic-csv sensor_data/sequence_2_*.csv \
  --haptic-offset 1.21 \
  --create-visualization --rotate-visualization
```

**独立生成（推荐用于调试）**：

```bash
# 基础可视化
python aria2lerobot/visualize.py \
  recording.vrs output_viz.mp4 \
  --max-frames 1000 --stride 3 --rotate

# 带触觉数据
python aria2lerobot/visualize_with_haptics.py \
  recording.vrs output_viz.mp4 \
  --haptic-csv sensor_data/sequence_2_*.csv \
  --haptic-offset 1.21 \
  --max-frames 1000
```

### 可视化参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `vrs_path` | VRS 文件路径（必需） | - |
| `output_path` | 输出视频路径（必需） | - |
| `--mps-dir` | MPS 目录 | 自动检测 |
| `--max-frames` | 最大帧数限制 | 无限制 |
| `--stride` | 帧间隔（1=每帧，3=每3帧） | 1 |
| `--rotate` | 顺时针旋转 90 度 | False |
| `--haptic-csv` | 触觉传感器 CSV 文件 | - |
| `--haptic-offset` | 时间偏移（秒） | 0.0 |
| `--max-force` | 最大力度值（用于归一化） | 150.0 |

---

## 数据模态说明

### observation.state (79 维)

| 维度范围 | 内容 | 描述 |
|---------|------|------|
| 0:63 | right_hand | 右手 21 个关键点 (21 x 3 坐标) |
| 63:65 | gaze | 眼睛注视点 UV 归一化坐标 [0-1] |
| 65:79 | device_pose | 设备位姿 (4x4 矩阵的 14 个元素) |

### action (8 维)

结构：`[left_total, right_total, left_xyz(3), right_xyz(3)]`

| 维度 | 内容 | CSV 来源 |
|-----|------|---------|
| 0 | 左手力合力大小 | Ring Tip_magnitude |
| 1 | 右手力合力大小 | Little Tip_magnitude |
| 2-4 | 左手力 XYZ 分量 | Ring Tip_fx/fy/fz |
| 5-7 | 右手力 XYZ 分量 | Little Tip_fx/fy/fz |

常用维度组合：
- 只要力的大小：`--action-dims 0,1`
- 只要左手完整数据：`--action-dims 0,2,3,4`
- 只要右手完整数据：`--action-dims 1,5,6,7`

### observation.images.main

1408 x 1408 x 3 RGB 视频帧（来自 Aria RGB 相机）。

### 手部关键点索引

右手 21 个关键点，每个点 3 个坐标 (x, y, z)，维度 = 关键点编号 x 3：

```
 0: WRIST           4: THUMB_TIP        8: INDEX_FINGER_TIP
12: MIDDLE_FINGER_TIP   16: RING_FINGER_TIP    20: PINKY_TIP
```

提取特定关键点示例（拇指尖 + 食指尖）：

```bash
python aria2lerobot/tools/extract_dimensions.py \
  output_glue output_glue_pinch \
  --state-dims 12,13,14,24,25,26 \
  --action-dims 0,1 \
  --state-name "pinch_fingertips" --action-name "force_magnitude"
```

---

## 参数说明

### 转换参数 (`python -m aria2lerobot`)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `vrs_path` | VRS 文件路径（必需） | - |
| `output_dir` | 输出目录（必需） | - |
| `--mps-dir` | MPS 目录路径 | 自动检测 |
| `--fps` | 目标 FPS | 自动检测 |
| `--max-slop-ns` | 时间对齐容差（纳秒） | 5ms |
| `--legacy-v21` | 使用 LeRobot v2.1 格式 | False |
| `--groot-format` | 生成 GR00T 兼容的 modality.json | False |
| `--stride` | 帧间隔 | 1 |
| `--max-frames` | 最大帧数 | 无限制 |
| `--overwrite` | 覆盖已存在的输出目录 | False |
| `--create-visualization` | 创建可视化视频 | False |
| `--rotate-visualization` | 旋转视频 90 度 | False |
| `--haptic-csv` | 触觉传感器 CSV 文件 | - |
| `--haptic-offset` | 时间偏移（秒） | 0.0 |
| `--split-csv` | 分段 CSV 文件 | - |
| `--task-name` | 任务名称 | "aria_recording" |
| `--task-description` | 任务描述 | - |
| `--log-level` | 日志级别 | INFO |
| `--quiet` | 静默模式 | False |

### 流水线脚本参数 (`aria_to_groot_pipeline.sh`)

| 参数 | 说明 |
|------|------|
| `--vrs` | VRS 文件路径（必需） |
| `--output` | 输出目录（必需） |
| `--haptic-csv` | 触觉传感器 CSV 文件 |
| `--haptic-offset` | 触觉数据时间偏移（秒） |
| `--split-csv` | 分段 CSV 文件 |
| `--task-name` | 任务名称 |
| `--skip-conversion` | 跳过转换，直接做 GR00T 格式验证 |
| `--test-loading` | 测试数据集加载 |

### 维度提取参数 (`extract_dimensions.py`)

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `source_dataset` | 源数据集目录（必需） | - |
| `output_dataset` | 输出数据集目录（必需） | - |
| `--state-dims` | State 维度（如 `0:63`, `3,7,12`） | 保留全部 |
| `--action-dims` | Action 维度（如 `0,1`, `2:5`） | 保留全部 |
| `--state-name` | State 模态名称 | extracted_state |
| `--action-name` | Action 模态名称 | extracted_action |
| `--format` | 输出格式：`groot`（训练）或 `lerobot`（可视化） | groot |

### 批量转换 (`batch_aria_to_groot.sh`)

编辑脚本中的 `TASKS` 数组配置多个任务：

```bash
declare -a TASKS=(
    "glue.vrs|output_glue|sensor_data/sequence_2_*.csv|1.21|aria_haptic_alignment/glue_alignment_segments.csv|pick and place glue|Assembly task using glue"
    "pen.vrs|output_pen|sensor_data/sequence_3_*.csv|1.13|aria_haptic_alignment/pen_alignment_segments.csv|pick and place pen|Assembly task using pen"
)
```

---

## 故障排除

### KeyError: 'chunk_index' 或 'episode_chunk'

info.json 中的路径模式使用了错误的变量名。确保 `aria2lerobot/schema.py` 中使用 GR00T 兼容格式：

```python
"data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
```

### KeyError: 'observation.state' in stats.json

stats.json 缺少 q01 和 q99 百分位数。重新计算：

```bash
python aria2lerobot/tools/compute_stats.py output_glue
```

### 数据集加载时 Segmentation fault

可能原因：内存不足、数据维度过大、视频解码问题。解决方案：
1. 使用维度提取工具减少数据维度
2. 减少 batch size
3. 使用 `--video-backend pyav` 代替 `decord`

### 想要左手数据而不是右手

修改 `aria2lerobot/schema.py` 中的 `flatten_hand_landmarks` 函数，将 `right_landmarks` 改为 `left_landmarks`，然后重新转换数据。

### 触觉数据时间不同步

调整 `--haptic-offset` 参数。使用 `aria_haptic_alignment` 工具确定正确的偏移量。

### 可视化视频中没有手部关键点

检查 MPS 目录中是否有 `hand_tracking.csv`，确保录制时手部在相机视野内。使用 `--verbose` 查看详细日志。

### 力度指示条显示为 0

触觉数据时间不同步。调整 `--haptic-offset`，检查 CSV 文件列名是否包含 `magnitude` 或 `fx/fy/fz`。

### 视频编码失败

OpenCV 编码器不可用。重新安装：

```bash
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python
```

工具会自动尝试多个编码器：H.264 → MP4V → AVC1 → X264。

---

## 参考文档

- [LeRobot v3.0](https://github.com/huggingface/lerobot)
- [NVIDIA GR00T](https://developer.nvidia.com/isaac-lab)
- [Project Aria](https://projectaria.com/)
- [MediaPipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)

## 更新日志

### 2025-11-24
- 添加完整的数据可视化章节（命令行 + 编程 API）
- 添加视频编码 API 文档（VideoEncoder、分片编码）

### 2025-10-03
- 创建 4 种数据集变体用于对照实验
- 添加捏合距离提取工具（1D state）和捏合特征提取工具（7D state）
- 添加 `--format` 参数支持 GR00T/LeRobot 格式切换

### 2025-10-02
- 修改代码存储**右手**而不是左手
- 修复 info.json 路径模式为 GR00T 兼容格式
- 添加 q01 和 q99 百分位数到 stats.json
- 创建维度提取工具用于实验对照组

### 2025-10-01
- 添加任务名称和描述参数
- 创建 GR00T 训练流程和批量处理脚本

---

**许可证**: MIT
