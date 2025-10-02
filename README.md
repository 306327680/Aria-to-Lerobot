# Aria2LeRobot - Project Aria到LeRobot数据集转换器

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-production--ready-brightgreen.svg)](https://github.com)

**将Project Aria VRS/MPS数据转换为标准LeRobot数据集格式,支持完整的多模态对齐、触觉数据集成和可视化功能。**

> **🎯 完全符合LeRobot v3.0格式规范** - 生成的数据集采用标准的`data/chunk-000/`和`videos/chunk-000/observation.images.main/`目录结构,使用Parquet元数据格式(tasks.parquet, episodes/),可直接被LeRobotDataset v0.3+加载和可视化。

> **🆕 触觉传感器支持** - 支持集成外部触觉传感器数据(CSV格式),自动时间对齐,并在可视化视频中显示力度指示条。详见[触觉数据集成指南](README_HAPTIC.md)和[快速开始](QUICK_START.md)。

基于可靠的`aria_api`构建,实现无缝数据访问,将复杂的Aria录制数据转换为标准化的LeRobot数据集,并提供带有手部关键点、眼动数据和触觉力度的可视化视频。

## ✨ 核心特性

### 🎯 **格式支持**
- **LeRobot v3.0格式**：完全符合官方LeRobot v3.0数据集规范
- **分块架构**：支持`chunk-000/`结构的大数据集
- **Parquet元数据**：v3.0标准格式 - `tasks.parquet`和`episodes/chunk-000/file-000.parquet`
- **向后兼容**：可选v2.1格式支持（使用`--legacy-v21`）
- 严格的模式验证和一致的数据类型

### 📊 **多模态数据**
- **视觉**：RGB相机帧 → H.264/H.265编码视频（10fps，正确颜色通道）
- **空间**：世界坐标系下的设备位姿（SE3变换）
- **手部**：世界坐标系下21点3D关键点（米为单位，Project Aria模型）
- **眼动**：注视点投影到相机平面
- **触觉**：支持外部触觉传感器CSV数据集成（自动时间对齐）
- **时间**：纳秒精度时间戳，设备时间对齐

### 🎥 **可视化功能**
- **手部关键点可视化**：21个关键点的骨骼连接（符合Project Aria规范）
- **彩虹手指颜色**：不同手指使用不同颜色（红橙黄绿蓝紫白）
- **眼动数据可视化**：注视点十字准星标记
- **触觉力度指示条**：实时显示左右手触觉传感器力度（带渐变色）
- **实时信息叠加**：帧号、时间戳、有效性统计
- **双手区分显示**：左手亮色，右手暗色
- **图像旋转支持**：可选顺时针90度旋转（仅影响可视化视频）
- **多Episode可视化**：支持为每个episode生成单独的可视化视频
- **独立可视化工具**：可单独生成可视化视频

### 🔄 **鲁棒处理**
- **时间对齐**：内置最近邻查询，可配置容差
- **触觉数据对齐**：支持自定义时间偏移和线性插值
- **坐标系统**：通过`aria_api`自动设备→世界变换
- **错误处理**：缺失数据时优雅降级，使用NaN值和有效性标志
- **内存高效**：流式处理支持大型数据集（<1GB内存使用）

### 🏗️ **生产就绪**
- **CLI界面**：简单的命令行使用，自动检测
- **增量写入**：Parquet分片，可扩展数据集
- **验证**：全面的输出验证和一致性检查
- **日志记录**：详细的进度跟踪和错误报告

## 🚀 快速开始

### 安装

```bash
# 安装依赖
pip install pyarrow pandas opencv-python tqdm orjson

# 确保aria_api可用（应在父目录中）
# 无需额外安装 - 直接使用！
```

### 基础使用

```bash
# ⚠️ 重要：必须从 aria2lerobot 的父目录执行
cd /Volumes/BU100/Aria_data

# 基础转换
python3 -m aria2lerobot glue.vrs output_dataset/

# 带可视化转换
python3 -m aria2lerobot glue.vrs output_dataset/ --create-visualization

# 带旋转可视化转换
python3 -m aria2lerobot glue.vrs output_dataset/ \
    --create-visualization \
    --rotate-visualization

# 完整命令：带触觉数据、可视化和旋转
python3 -m aria2lerobot glue.vrs output \
    --haptic-csv sensor_data/sequence_2_20250927_203243.csv \
    --haptic-offset 1.31 \
    --create-visualization \
    --rotate-visualization \
    --overwrite
```

### 触觉数据集成

**使用触觉传感器独立工具集**：

```bash
# 1. 快速测试（推荐）
cd /Volumes/BU100/Aria_data/aria2lerobot
./test_haptic_integration.sh

# 2. 生成带触觉力度指示条的可视化视频
python3 visualize_with_haptics.py \
    ../glue.vrs output_video.mp4 \
    --haptic-csv ../sensor_data/sequence_2_20250927_203243.csv \
    --haptic-offset 1.31 \
    --max-force 150

# 3. 生成包含触觉数据的Parquet数据集
python3 convert_with_haptics.py \
    ../glue.vrs output_dataset/ \
    --haptic-csv ../sensor_data/sequence_2_20250927_203243.csv \
    --haptic-offset 1.31 \
    --fps 10
```

**详细文档**：
- [触觉数据集成完整指南](README_HAPTIC.md)
- [触觉功能快速开始](QUICK_START.md)

### Python API使用

```python
from aria2lerobot import convert

# 基础转换
result = convert(
    vrs_path="glue.vrs",
    mps_dir="mps_glue_vrs",
    out_dir="output_dataset"
)

# 高级选项（含触觉数据）
result = convert(
    vrs_path="glue.vrs",
    mps_dir="mps_glue_vrs",
    out_dir="output_dataset",
    fps=15.0,                      # 自定义FPS
    max_slop_ns=10_000_000,       # 10ms对齐容差
    stride=2,                      # 每隔一帧
    max_frames=50000,             # 限制处理帧数
    overwrite=True,               # 覆盖现有输出
    create_visualization=True,    # 生成可视化视频
    rotate_visualization=True,    # 旋转可视化视频90度
    haptic_csv="sensor.csv",      # 触觉数据CSV
    haptic_offset=1.31            # 触觉时间偏移（秒）
)

print(f"处理了 {result['processing']['valid_frames']} 帧")
print(f"手部有效率: {result['data_quality']['hand_valid_rate']:.1%}")
```

## 📁 输出结构

### LeRobot v3.0格式（符合官方规范）

```
output_dataset/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet    # 剧集表格数据
│       ├── episode_000001.parquet    # 多剧集支持
│       └── ...
├── videos/
│   └── chunk-000/
│       └── observation.images.main/  # 相机名称目录
│           ├── episode_000000.mp4    # 剧集视频
│           ├── episode_000001.mp4
│           └── ...
├── meta/
│   ├── info.json                     # 模式定义、FPS、剧集信息
│   ├── stats.json                    # 数据统计（空文件，由LeRobot自动计算）
│   ├── tasks.parquet                 # 任务描述（v3.0 Parquet格式）
│   └── episodes/                     # 剧集元数据目录（v3.0新格式）
│       └── chunk-000/
│           └── file-000.parquet      # 剧集元数据（Parquet格式）
├── visualization_episode_000000.mp4  # Episode 0可视化视频（可选，使用--create-visualization）
├── visualization_episode_000001.mp4  # Episode 1可视化视频（多episode模式）
├── ...                               # 更多episode可视化视频
└── conversion.log                    # 详细转换日志
```

### 关键格式特性（v3.0）

- **📂 分块结构**：`chunk-000/`目录支持大数据集分块
- **📋 Parquet元数据**：v3.0使用`tasks.parquet`和`episodes/chunk-000/file-000.parquet`替代JSONL
- **🎥 相机分组**：视频按相机类型组织（`observation.images.main/`）
- **📊 剧集导向**：每个剧集一个parquet和mp4文件对
- **🔄 LeRobot v0.3+兼容**：可直接使用`lerobot-dataset-viz`可视化

## 🔄 Aria到LeRobot数据映射

### 数据源对应关系

| Aria数据源 | LeRobot列名 | 数据类型 | 说明 |
|------------|-------------|----------|------|
| **相机数据** |
| `camera-rgb`帧 | `observation.images.main` | 视频(MP4) | RGB相机帧，H.264编码 |
| **位姿数据** |
| MPS闭环轨迹 | `status.pose.world_device` | float32[16] | 世界→设备4×4变换矩阵(扁平化) |
| 位姿有效性 | `status.pose.valid` | bool | 位姿数据是否有效 |
| **手部跟踪** |
| 手部21关键点(世界坐标) | `status.hands.landmarks_world` | float32[63] | 21×3关键点坐标扁平化(米) |
| 左手有效性 | `status.hands.valid_left` | bool | 左手跟踪是否有效 |
| 右手有效性 | `status.hands.valid_right` | bool | 右手跟踪是否有效 |
| **标准LeRobot列** |
| 手部关键点 + 设备位姿 | `observation.state` | float32[79] | 观测向量：手部(63) + 位姿(16) |
| 触觉力度数据 | `action` | float32[8] | 动作向量：双手触觉力度（见下文详细说明） |
| **触觉数据（详细列）** |
| 外部触觉传感器CSV | `action.tactile.left.force_total` | float32 | 左手总力（需外部数据） |
| 外部触觉传感器CSV | `action.tactile.right.force_total` | float32 | 右手总力（需外部数据） |
| 外部触觉传感器CSV | `action.tactile.left.force_xyz` | float32[3] | 左手力向量[fx, fy, fz]（需外部数据） |
| 外部触觉传感器CSV | `action.tactile.right.force_xyz` | float32[3] | 右手力向量[fx, fy, fz]（需外部数据） |
| **触觉掩码** |
| N/A | `action.mask.tactile.left.force_total` | bool | 左手总力掩码 |
| N/A | `action.mask.tactile.right.force_total` | bool | 右手总力掩码 |
| N/A | `action.mask.tactile.left.force_xyz` | bool[3] | 左手力向量掩码 |
| N/A | `action.mask.tactile.right.force_xyz` | bool[3] | 右手力向量掩码 |

### 重要说明

#### 📊 **`action`向量详细结构（8维）**

`action`向量包含双手触觉传感器的力度数据，用于神经网络训练（输入`observation.state`预测`action`）：

| 位置 | 名称 | 说明 | 单位 |
|------|------|------|------|
| 0 | `left_total` | 左手传感器总力（合力大小） | 牛顿(N) |
| 1 | `right_total` | 右手传感器总力（合力大小） | 牛顿(N) |
| 2 | `left_fx` | 左手X方向分力 | 牛顿(N) |
| 3 | `left_fy` | 左手Y方向分力 | 牛顿(N) |
| 4 | `left_fz` | 左手Z方向分力 | 牛顿(N) |
| 5 | `right_fx` | 右手X方向分力 | 牛顿(N) |
| 6 | `right_fy` | 右手Y方向分力 | 牛顿(N) |
| 7 | `right_fz` | 右手Z方向分力 | 牛顿(N) |

**数据来源**：
- 从外部触觉传感器CSV文件中提取（`--haptic-csv`参数）
- CSV文件需包含列：`Ring Tip_fx/fy/fz/magnitude`, `Little Tip_fx/fy/fz/magnitude`
- 通过线性插值对齐到每个视频帧的时间戳
- 使用`--haptic-offset`参数调整时间偏移以优化对齐精度

**使用示例**：
```python
# 训练示例：预测触觉力度
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import torch

dataset = LeRobotDataset("output_dataset")
sample = dataset[0]

# 输入：观测状态（手部关键点 + 设备位姿）
observation = sample['observation.state']  # shape: (79,)

# 输出：触觉力度（双手力度向量）
action = sample['action']  # shape: (8,)

# 解析action向量
left_total = action[0]      # 左手总力
right_total = action[1]     # 右手总力
left_xyz = action[2:5]      # 左手xyz分量
right_xyz = action[5:8]     # 右手xyz分量

print(f"左手力度: {left_total:.2f}N [fx={left_xyz[0]:.2f}, fy={left_xyz[1]:.2f}, fz={left_xyz[2]:.2f}]")
print(f"右手力度: {right_total:.2f}N [fx={right_xyz[0]:.2f}, fy={right_xyz[1]:.2f}, fz={right_xyz[2]:.2f}]")
```

#### 🤚 **触觉数据处理**
- **Project Aria设备不含触觉传感器**：设备本身无触觉硬件
- **外部触觉数据支持**：支持集成外部触觉传感器CSV数据
  - 通过`--haptic-csv`参数指定CSV文件
  - 通过`--haptic-offset`参数调整时间对齐（秒）
  - 使用线性插值实现帧级对齐
  - 自动提取xyz分力和总力（magnitude）
  - 专用工具：`convert_with_haptics.py`和`visualize_with_haptics.py`
- **LeRobot兼容性**：无触觉数据时填充NaN，掩码设为False
- **可视化支持**：可在视频中显示力度指示条（渐变色，左右手分开）

#### 🌍 **坐标系统**
- **设备坐标系**：以左眼SLAM相机为原点
- **世界坐标系**：MPS重建的重力对齐世界坐标系
- **变换处理**：所有坐标变换由`aria_api`自动处理
- **单位统一**：位置使用米(m)，角度使用弧度(rad)

#### ⏰ **时间对齐**
- **主时间线**：以`camera-rgb`帧时间戳为基准
- **对齐方式**：最近邻查询，默认5ms容差
- **触觉对齐**：支持自定义偏移+线性插值
- **缺失处理**：超出容差的数据填充NaN，设置有效性标志为False

## 📊 数据模式

### 核心列

| 列名 | 类型 | 形状 | 描述 |
|--------|------|-------|-------------|
| `timestamp` | float64 | () | 时间戳（秒） |
| `timestamp_ns` | int64 | () | 时间戳（纳秒，设备时间） |
| `episode_id` | int32 | () | 剧集标识符 |
| `step` | int64 | () | 剧集内步骤（从0开始） |

### 手部数据

| 列名 | 类型 | 形状 | 描述 |
|--------|------|-------|-------------|
| `status.hands.landmarks_world` | float32 | (63,) | 世界坐标系下手部关键点（21×3扁平化） |
| `status.hands.valid_left` | bool | () | 左手有效性标志 |
| `status.hands.valid_right` | bool | () | 右手有效性标志 |

### 位姿数据

| 列名 | 类型 | 形状 | 描述 |
|--------|------|-------|-------------|
| `status.pose.world_device` | float32 | (16,) | 世界→设备变换（4×4矩阵扁平化） |
| `status.pose.valid` | bool | () | 位姿有效性标志 |

### 触觉动作

> **⚠️ 重要说明**：Project Aria设备不包含触觉传感器。可通过外部CSV文件集成触觉数据，否则数据为NaN。

| 列名 | 类型 | 形状 | 描述 |
|--------|------|-------|-------------|
| `action.tactile.left.force_total` | float32 | () | 左手总力（需外部数据） |
| `action.tactile.right.force_total` | float32 | () | 右手总力（需外部数据） |
| `action.tactile.left.force_xyz` | float32 | (3,) | 左手力向量（需外部数据） |
| `action.tactile.right.force_xyz` | float32 | (3,) | 右手力向量（需外部数据） |
| `action.mask.tactile.*` | bool | () | 对应掩码（有数据时为True） |

### 视频数据

| 列名 | 描述 |
|--------|-------------|
| `observation.images.main` | 存储为MP4视频的RGB相机帧 |

## 🔧 配置选项

### 命令行参数完整列表

| 参数 | 类型 | 必须/可选 | 默认值 | 说明 |
|------|------|-----------|--------|------|
| **基础参数** |
| `vrs_path` | 位置参数 | ✅ 必须 | - | VRS文件路径 |
| `output_dir` | 位置参数 | ✅ 必须 | - | 输出目录路径 |
| `--mps-dir` | 字符串 | 可选 | 自动检测 | MPS目录路径（若不指定则自动检测） |
| **时间与采样** |
| `--fps` | 浮点数 | 可选 | 自动检测 | 目标帧率（Hz） |
| `--max-slop-ns` | 整数 | 可选 | 5000000 | 时间对齐容差（纳秒，默认5ms） |
| `--stride` | 整数 | 可选 | 1 | 帧步长（1=每帧，2=隔帧，3=每三帧） |
| `--max-frames` | 整数 | 可选 | 无限制 | 最大处理帧数（用于测试或预览） |
| **格式选项** |
| `--legacy-v21` | 标志 | 可选 | False | 使用LeRobot v2.1格式（默认v3.0） |
| `--overwrite` | 标志 | 可选 | False | 覆盖已存在的输出目录 |
| **剧集分割** |
| `--split-csv` | 字符串 | 可选 | - | CSV文件路径，包含episode分割时间点（split_time_s列） |
| **可视化** |
| `--create-visualization` | 标志 | 可选 | False | 生成带手部关键点和眼动的可视化视频（多episode时为每个episode生成） |
| `--rotate-visualization` | 标志 | 可选 | False | 将可视化视频旋转90度（仅影响可视化） |
| **触觉数据** |
| `--haptic-csv` | 字符串 | 可选 | - | 触觉传感器CSV文件路径 |
| `--haptic-offset` | 浮点数 | 可选 | 0.0 | 触觉数据时间偏移（秒） |
| **日志与调试** |
| `--log-level` | 枚举 | 可选 | INFO | 日志级别：DEBUG/INFO/WARNING/ERROR |
| `--quiet` | 标志 | 可选 | False | 抑制进度输出 |

### 参数使用示例

```bash
# 查看所有参数
python3 -m aria2lerobot --help

# 最简单的命令（必需参数）
python3 -m aria2lerobot glue.vrs output/

# 完整参数示例
python3 -m aria2lerobot glue.vrs output/ \
    --mps-dir mps_glue_vrs \
    --fps 10 \
    --max-slop-ns 10000000 \
    --stride 1 \
    --max-frames 1000 \
    --create-visualization \
    --rotate-visualization \
    --haptic-csv sensor_data/data.csv \
    --haptic-offset 1.31 \
    --overwrite \
    --log-level DEBUG
```

### 可视化功能详解

转换器支持生成包含手部关键点、眼动数据和触觉力度的可视化视频：

#### 🎨 可视化内容
- **手部骨骼**：21个关键点连接线（符合Project Aria规范）
  - 拇指：手腕(5) → 中间(6) → 远端(7) → 指尖(0) - **红色**
  - 食指：手腕(5) → 近端(8) → 中间(9) → 远端(10) → 指尖(1) - **橙色**
  - 中指：手腕(5) → 近端(11) → 中间(12) → 远端(13) → 指尖(2) - **黄色**
  - 无名指：手腕(5) → 近端(14) → 中间(15) → 远端(16) → 指尖(3) - **绿色**
  - 小指：手腕(5) → 近端(17) → 中间(18) → 远端(19) → 指尖(4) - **蓝色**
  - 手掌：手腕(5) ↔ 手掌中心(20) ↔ 各手指根部 - **白色/紫色**
- **眼动数据**：红色十字准星标记注视点
- **触觉力度**：右侧力度指示条（渐变色：绿→黄→红）
  - 左手传感器：L标签
  - 右手传感器：R标签
  - 实时数值显示
- **颜色编码**：左手亮色，右手深色（便于区分）
- **图像旋转**：可选顺时针90度旋转（仅影响可视化视频，不影响数据集）
- **信息叠加**：帧号、时间戳、有效性统计

#### 🔧 可视化参数
```bash
# 集成到转换过程
--create-visualization   # 在转换同时生成可视化视频
--rotate-visualization   # 旋转可视化视频90度（仅影响可视化）
--haptic-csv PATH       # 触觉数据CSV文件

# 独立可视化工具参数
--max-frames INT        # 最大处理帧数
--stride INT           # 帧步长
--rotate               # 旋转图像90度
--max-force FLOAT      # 触觉力度最大值（用于归一化）
--verbose              # 详细日志输出
```

### 时间对齐

转换器使用**camera-rgb帧作为主要时间线**：

- 对每个相机帧时间戳，查询对齐的数据：
  - **位姿**：`session.nearest_pose(ts_ns, max_slop_ns)`（优先闭环）
  - **手部**：`session.hands.landmarks_in_world(ts_ns, T_world_device, max_slop_ns)`
  - **触觉**：线性插值到帧时间戳（可配置偏移）
- 默认容差：**5ms**（`max_slop_ns=5_000_000`）
- 触觉对齐：使用`aria_haptic_alignment`计算最优偏移
- 缺失数据 → NaN值 + 有效性标志 = False

### 坐标系统

- **设备坐标系**：左SLAM相机坐标系
- **世界坐标系**：MPS闭环重力对齐世界坐标
- **单位**：位置用米，角度用弧度
- **变换**：由`aria_api`自动处理

## 📈 性能与可扩展性

### 基准测试

基于典型Aria录制：

| 数据集大小 | 帧数 | 处理时间 | 内存使用 | 输出大小 |
|-------------|--------|-----------------|--------------|-------------|
| 短（5分钟） | ~9,000 | 30秒 | <500MB | ~2GB |
| 中（30分钟） | ~54,000 | 3分钟 | <800MB | ~12GB |
| 长（2小时） | ~216,000 | 12分钟 | <1GB | ~48GB |

### 内存效率

- **流式处理**：增量处理帧
- **Parquet分片**：默认每片50k行
- **视频编码**：直接帧到MP4，无缓冲
- **大数据集支持**：已测试>1M帧

## 🧪 使用示例

### 基础转换

```bash
# ⚠️ 从父目录执行（重要！）
cd /Volumes/BU100/Aria_data

# 转换glue.vrs，自动检测MPS
python3 -m aria2lerobot glue.vrs glue_dataset/ --overwrite

# 带可视化转换
python3 -m aria2lerobot glue.vrs glue_dataset/ \
    --create-visualization \
    --overwrite

# 带旋转可视化转换
python3 -m aria2lerobot glue.vrs glue_dataset/ \
    --create-visualization \
    --rotate-visualization \
    --overwrite

# 完整命令：带触觉数据、可视化和旋转
python3 -m aria2lerobot glue.vrs glue_dataset \
    --haptic-csv sensor_data/sequence_2_20250927_203243.csv \
    --haptic-offset 1.31 \
    --create-visualization \
    --rotate-visualization \
    --overwrite
```

预期输出：
```
🚀 Aria to LeRobot 转换
📁 VRS文件: glue.vrs
📂 MPS目录: mps_glue_vrs
📤 输出: glue_dataset/
🎯 格式: 标准LeRobot格式
📊 触觉数据: sensor_data.csv (偏移: 1.31s)

转换帧数: 100%|████████| 18750/18750 [02:15<00:00, 138.2帧/秒]

✅ 转换成功完成！
📊 处理: 18,750 帧
⏱️  时间: 135.2秒
🗂️  文件: 1个parquet + 2个视频
📈 手部有效率: 87.3%
📈 位姿有效率: 99.9%
```

### 触觉数据集成示例

```bash
# 1. 计算时间偏移（使用aria_haptic_alignment）
cd /Volumes/BU100/Aria_data/aria_haptic_alignment
python3 aria_haptic_align.py
# 输出: Optimal offset: 1.310 seconds

# 2. 完整转换命令（从父目录执行）
cd /Volumes/BU100/Aria_data
python3 -m aria2lerobot glue.vrs output \
    --haptic-csv sensor_data/sequence_2_20250927_203243.csv \
    --haptic-offset 1.31 \
    --create-visualization \
    --rotate-visualization \
    --overwrite

# 3. 使用独立触觉可视化工具（从 aria2lerobot 目录执行）
cd /Volumes/BU100/Aria_data/aria2lerobot
python3 visualize_with_haptics.py ../glue.vrs visualization.mp4 \
    --haptic-csv ../sensor_data/sequence_2_20250927_203243.csv \
    --haptic-offset 1.31 \
    --max-force 150 \
    --max-frames 300
```

### 多Episode数据集转换与可视化

```bash
# 从父目录执行
cd /media/echo/BU100/Aria_data

# 使用CSV分割点创建多episode数据集，并为每个episode生成可视化视频
python3 -m aria2lerobot pen.vrs output_pen \
    --haptic-csv sensor_data/sequence_3_20250927_203243.csv \
    --haptic-offset 1.15 \
    --split-csv aria_haptic_alignment/pen_alignment_segments.csv \
    --create-visualization \
    --rotate-visualization \
    --overwrite

# CSV格式示例 (pen_alignment_segments.csv):
# split_time_s
# 6.2
# 10.4
# 14.8
# ... (每行一个分割时间点，单位：秒)

# 输出：
# - 45个episode的parquet文件: episode_000000.parquet ~ episode_000044.parquet
# - 45个episode的视频文件: episode_000000.mp4 ~ episode_000044.mp4
# - 45个可视化视频: visualization_episode_000000.mp4 ~ visualization_episode_000044.mp4
#   (每个视频显示对应episode的正确帧和触觉力度)
```

### 自定义采样

```bash
# ⚠️ 从父目录执行
cd /Volumes/BU100/Aria_data

# 高质量转换带可视化
python3 -m aria2lerobot glue.vrs output \
    --fps 30 \
    --stride 1 \
    --create-visualization

# 高质量转换带旋转可视化
python3 -m aria2lerobot glue.vrs output \
    --fps 30 \
    --stride 1 \
    --create-visualization \
    --rotate-visualization

# 快速预览（少量帧数）
python3 -m aria2lerobot glue.vrs preview \
    --stride 20 \
    --max-frames 50 \
    --create-visualization
```

### 独立可视化工具

```bash
# 在 aria2lerobot 目录内执行独立工具
cd /Volumes/BU100/Aria_data/aria2lerobot

# 标准可视化（不带触觉）
python3 visualize.py ../glue.vrs visualization_output.mp4 \
    --max-frames 100 \
    --stride 10

# 旋转可视化
python3 visualize.py ../glue.vrs rotated_output.mp4 \
    --max-frames 100 \
    --stride 10 \
    --rotate

# 带触觉数据的完整可视化
python3 visualize_with_haptics.py ../glue.vrs output.mp4 \
    --haptic-csv ../sensor_data/sequence_2_20250927_203243.csv \
    --haptic-offset 1.31 \
    --max-force 150 \
    --max-frames 300
```

### 用LeRobot读取

```python
# 转换后，用LeRobot加载
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# 加载数据集
dataset = LeRobotDataset("output_dataset")

print(f"数据集: {len(dataset)} 帧")
print(f"列: {list(dataset.features.keys())}")

# 访问数据
sample = dataset[0]
print(f"时间戳: {sample['timestamp']:.3f}秒")
print(f"手部关键点形状: {sample['status.hands.landmarks_world'].shape}")
print(f"视频帧形状: {sample['observation.images.main'].shape}")

# 检查有效性
if sample['status.hands.valid_left']:
    landmarks = sample['status.hands.landmarks_world'].reshape(21, 3)
    print(f"左手手腕位置: {landmarks[20]}")  # 手腕是第20个点

# 检查触觉数据
import numpy as np
tactile_force = sample['action.tactile.left.force_total']
tactile_mask = sample['action.mask.tactile.left.force_total']
if tactile_mask and not np.isnan(tactile_force):
    print(f"左手力度: {tactile_force:.2f}N")
else:
    print(f"无触觉数据 (NaN: {np.isnan(tactile_force)})")
```

### 批量处理

```python
import glob
from aria2lerobot import convert

# 处理目录中所有VRS文件
vrs_files = glob.glob("*.vrs")
for vrs_file in vrs_files:
    output_dir = f"datasets/{Path(vrs_file).stem}_lerobot"

    try:
        result = convert(vrs_file, None, output_dir, overwrite=True)
        print(f"✅ {vrs_file}: {result['processing']['valid_frames']} 帧")
    except Exception as e:
        print(f"❌ {vrs_file}: {e}")
```

## 🔍 数据质量与验证

### 质量指标

转换器跟踪并报告：

- **帧丢失率**：无法处理的相机帧百分比
- **手部有效率**：有效手部跟踪的帧百分比
- **位姿有效率**：有效设备位姿的帧百分比
- **触觉有效率**：有效触觉数据的帧百分比
- **NaN率**：每列NaN出现率
- **时间对齐**：对齐误差分布

### 验证检查

转换后验证包括：

- **模式合规**：所有列匹配预期类型/形状
- **文件一致性**：视频帧数与parquet行数匹配
- **时间连续性**：无大的时间戳间隙
- **数据范围**：值在合理物理范围内

### 故障排除

#### 常见问题

**1. "未找到MPS目录"**
```bash
# 解决方案：显式指定MPS目录
python3 -m aria2lerobot data.vrs output/ --mps-dir path/to/mps_dir
```

**2. "高帧丢失率(>10%)"**
```bash
# 解决方案：增加对齐容差
python3 -m aria2lerobot data.vrs output/ --max-slop-ns 10000000  # 10ms
```

**3. "低手部有效率(<50%)"**
- 检查MPS手部跟踪质量
- 验证hand_tracking_results.csv存在且有数据
- 考虑不同录制条件

**4. "内存使用过高"**
- 使用帧步长：`--stride 2`
- 限制处理：`--max-frames 10000`
- 检查可用系统内存

**5. "触觉数据全是NaN"**
- 未提供`--haptic-csv`参数：Project Aria设备不含触觉传感器
- 检查CSV文件路径是否正确
- 验证时间偏移参数`--haptic-offset`
- 使用`aria_haptic_alignment`重新计算偏移

**6. "触觉时间对齐不准确"**
- 使用`aria_haptic_alignment`库计算最优偏移
- 微调`--haptic-offset`参数（±0.1秒）
- 检查CSV的`relative_time`列是否从0开始

**7. "视频FPS不正确或颜色反转"**
- 已修复：FPS估算现在使用stride=1而非stride=10
- 已修复：VRS RGB图像不再错误转换为BGR
- 视频现在以正确的10fps和RGB颜色通道编码

**8. "可视化视频中手部和眼动显示为N/A"**
- 已修复：正确解析aria_api返回的元组格式
- 手部投影：`(landmarks_2d, valid_mask)`
- 眼动投影：`(pixel_coords, depth, is_valid)`
- 手部连接现在符合Project Aria 21点模型规范
- 已添加彩虹色手指区分和畸变校正

**9. "投影精度问题（畸变校正）"**
- 已修复：集成Project Aria官方`CameraCalibration.project()`方法
- 使用畸变感知投影替代简单针孔模型
- 显著提高边缘区域的投影精度（差异可达0.2像素）
- 自动检测并应用相机畸变参数

**10. "可视化视频需要旋转90度"**
- 已实现：可选顺时针90度图像旋转功能
- 仅影响可视化视频，不影响LeRobot数据集
- 实现方式：在原始图像上投影手部和眼动数据，然后旋转整个帧
- 使用`cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)`进行旋转
- CLI参数：`--rotate-visualization`（集成）或`--rotate`（独立）
- 手部关键点和眼动数据在旋转后完美对齐

**11. "多episode可视化视频内容重复/触觉数据错误"**
- 已修复：每个episode的可视化视频现在从正确的VRS帧开始
- 已修复：触觉数据使用全局起始时间戳进行正确对齐
- 问题原因：
  - 原先查找起始帧时受`max_frames`限制，导致后续episode找不到正确起始帧
  - 触觉数据使用episode相对时间而非全局相对时间查询
- 修复方案（visualize.py:346-369, 491）：
  - 先在所有帧中查找`start_timestamp_ns`对应的起始帧
  - 再根据`max_frames`限制处理帧数
  - 添加`global_start_timestamp_ns`参数用于触觉数据全局时间对齐
  - 触觉查询使用`(timestamp_ns - global_start_timestamp_ns)`而非`(timestamp_ns - start_timestamp_ns)`
- 结果：45个episode的可视化视频正确显示不同内容和触觉力度

#### 调试模式

```bash
# 启用详细日志
python3 -m aria2lerobot data.vrs output/ --log-level DEBUG

# 检查conversion.log获取详细信息
tail -f output/conversion.log
```

## 🧪 测试与验证

### 烟雾测试

```bash
# 运行包含示例的烟雾测试
cd examples/
python3 smoke_test.py ../glue.vrs
```

预期输出演示：
- 成功转换为LeRobot格式
- 用LeRobotDataset加载数据
- 样本数据检查和验证

### 单元测试

```bash
# 运行单元测试
python3 -m pytest tests/ -v

# 测试特定功能
python3 -m pytest tests/test_schema.py -v
python3 -m pytest tests/test_conversion.py -v
```

### 触觉集成测试

```bash
# 运行触觉数据集成测试脚本
cd aria2lerobot
./test_haptic_integration.sh

# 预期输出：
# - Parquet数据集（含触觉列）
# - 可视化视频（含力度指示条）
# - 验证报告
```

## ✅ LeRobot v3.0格式兼容性

### 格式规范符合性

aria2lerobot完全符合官方LeRobot v3.0数据集格式规范，可直接被 `LeRobotDataset` v0.3+加载：

```
✅ 目录结构: data/chunk-000/, videos/chunk-000/observation.images.main/
✅ 文件命名: episode_000000.parquet, episode_000000.mp4 (标准格式)
✅ 元数据格式: tasks.parquet, episodes/chunk-000/file-000.parquet (v3.0 Parquet格式)
✅ info.json字段: codebase_version="v3.0", data_path, video_path, features
✅ 必需列: index, episode_index, episode_chunk, frame_index (LeRobot v3.0)
✅ 数据路径模板: chunk-{chunk_index:03d}/episode_{file_index:06d}.parquet
✅ 视频路径模板: chunk-{chunk_index:03d}/{video_key}/episode_{file_index:06d}.mp4
✅ 数据类型: 严格的PyArrow schema验证
✅ 视频编码: H.264/mp4v, RGB色彩空间
✅ 压缩格式: Zstandard压缩的Parquet文件
✅ 触觉数据: 兼容外部传感器CSV集成
✅ 可视化兼容: lerobot-dataset-viz完全支持
```

### LeRobotDataset 加载测试

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# 直接加载转换后的数据集
dataset = LeRobotDataset(
    repo_id="output",
    root="/path/to/aria2lerobot/output"
)

print(f"数据集大小: {len(dataset)}")
print(f"特征列: {list(dataset.features.keys())}")

# 访问数据
sample = dataset[0]
print(f"Index: {sample['index']}")
print(f"Episode: {sample['episode_index']}")
print(f"Frame: {sample['frame_index']}")
```

### 可视化工具兼容

```bash
# 使用 LeRobot v0.3+ 官方可视化工具
lerobot-dataset-viz \
    --repo-id output_dataset \
    --root /path/to/converted \
    --episode-index 0

# 或使用完整路径
python -m lerobot.scripts.lerobot_dataset_viz \
    --repo-id output_dataset \
    --root /path/to/converted \
    --episode-index 0
```

### 与现有LeRobot v0.3+工具兼容

生成的数据集可直接用于：
- **LeRobotDataset v0.3+**: 标准数据加载器
- **lerobot-dataset-viz**: 官方数据集可视化工具
- **Hugging Face Hub**: 数据集上传和共享
- **LeRobot训练脚本**: 端到端策略训练

⚠️ **重要提示**:
- v3.0格式数据集需要LeRobot v0.3或更高版本
- 如需与旧版LeRobot兼容，使用`--legacy-v21`参数生成v2.1格式

## 🤝 与LeRobot集成

### 数据集加载

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

# 加载转换的数据集
dataset = LeRobotDataset("path/to/converted/dataset")

# 与DataLoader一起使用
from torch.utils.data import DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in dataloader:
    images = batch['observation.images.main']  # (B, C, H, W)
    hands = batch['status.hands.landmarks_world']  # (B, 63)
    poses = batch['status.pose.world_device']  # (B, 16)
    tactile_left = batch['action.tactile.left.force_total']  # (B,)
    # ... 训练你的模型
```

### 策略训练示例

```python
# 示例：在手部位置和触觉上训练简单策略
import torch
import torch.nn as nn

class HandTactilePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.hand_encoder = nn.Linear(63, 128)  # 手部关键点
        self.pose_encoder = nn.Linear(16, 128)  # 设备位姿
        self.tactile_encoder = nn.Linear(2, 64)  # 触觉力度
        self.policy_head = nn.Linear(320, 6)    # 6自由度动作

    def forward(self, hands, poses, tactile):
        hand_feat = torch.relu(self.hand_encoder(hands))
        pose_feat = torch.relu(self.pose_encoder(poses))
        tactile_feat = torch.relu(self.tactile_encoder(tactile))
        combined = torch.cat([hand_feat, pose_feat, tactile_feat], dim=-1)
        return self.policy_head(combined)

# 训练循环
model = HandTactilePolicy()
dataset = LeRobotDataset("converted_dataset")
dataloader = DataLoader(dataset, batch_size=32)

for batch in dataloader:
    hands = batch['status.hands.landmarks_world']
    poses = batch['status.pose.world_device']
    tactile = torch.stack([
        batch['action.tactile.left.force_total'],
        batch['action.tactile.right.force_total']
    ], dim=-1)

    actions = model(hands, poses, tactile)
    # ... 计算损失和反向传播
```

## 📚 技术细节

### 数据转换流程

#### 核心转换逻辑（convert.py）

**1. HapticDataLoader类（触觉数据加载）**
```python
class HapticDataLoader:
    """从CSV文件加载触觉传感器数据并进行时间对齐"""

    def __init__(self, csv_path: str, time_offset: float = 0.0):
        # 加载CSV文件
        df = pd.read_csv(csv_path)

        # 提取时间戳（relative_time列）
        self.timestamps = df['relative_time'].values

        # 提取双手传感器的magnitude和xyz分力
        # 例如：Ring Tip_magnitude, Ring Tip_fx, Ring Tip_fy, Ring Tip_fz
        self.force1_magnitude = df['Ring Tip_magnitude'].values
        self.force1_xyz = np.stack([
            df['Ring Tip_fx'].values,
            df['Ring Tip_fy'].values,
            df['Ring Tip_fz'].values
        ], axis=1)  # shape: (N, 3)

        # 同理处理第二个传感器（Little Tip）
        self.force2_magnitude = df['Little Tip_magnitude'].values
        self.force2_xyz = np.stack([...], axis=1)

    def get_force_at_time(self, time_s: float) -> Tuple[Tuple[float, np.ndarray], Tuple[float, np.ndarray]]:
        """线性插值获取指定时间的力度值"""
        # 应用时间偏移
        query_time = time_s - self.time_offset

        # 边界检查
        if query_time < self.timestamps[0]:
            return (0.0, np.zeros(3)), (0.0, np.zeros(3))

        # 查找插值索引
        idx = np.searchsorted(self.timestamps, query_time)

        # 线性插值magnitude和xyz分量
        alpha = (query_time - t0) / (t1 - t0)
        force1_mag = f1_mag_0 + alpha * (f1_mag_1 - f1_mag_0)
        force1_xyz = f1_xyz_0 + alpha * (f1_xyz_1 - f1_xyz_0)

        return (force1_mag, force1_xyz), (force2_mag, force2_xyz)
```

**2. 帧数据提取（_extract_frame_data函数）**
```python
def _extract_frame_data(session, timestamp_ns, step, episode_id, ...):
    """提取单帧的所有对齐数据"""

    # 1. 提取位姿数据
    pose = session.nearest_pose(timestamp_ns, max_slop_ns)
    if pose is not None:
        pose_matrix = pose.T_world_device.to_matrix()  # 4x4矩阵
        row_data["status.pose.world_device"] = flatten_pose_matrix(pose_matrix)  # 扁平化为16维

    # 2. 提取手部关键点（世界坐标）
    hands_data = session.nearest_hands(timestamp_ns, max_slop_ns)
    if hands_data and pose:
        world_landmarks = session.hand_projector.landmarks_in_world(
            timestamp_ns, pose.T_world_device, max_slop_ns
        )
        left_landmarks = world_landmarks.get("left")   # shape: (21, 3)
        right_landmarks = world_landmarks.get("right")

        # 扁平化为63维向量（只保存左手，或可扩展保存双手）
        flattened = flatten_hand_landmarks(left_landmarks, right_landmarks)
        row_data["status.hands.landmarks_world"] = flattened  # shape: (63,)

    # 3. 提取触觉力度（线性插值到当前帧时间）
    time_s = (timestamp_ns - start_timestamp_ns) / 1e9
    if haptic_loader:
        force_left, force_right = haptic_loader.get_force_at_time(time_s)
        # force_left = (magnitude: float, xyz: np.ndarray[3])
        # force_right = (magnitude: float, xyz: np.ndarray[3])

    # 4. 创建触觉数据列
    tactile_data = create_tactile_placeholders(force_left, force_right)
    row_data.update(tactile_data)
    # 生成列：
    # - action.tactile.left.force_total (总力)
    # - action.tactile.left.force_xyz (xyz分量)
    # - action.tactile.right.* (同理)
    # - action.mask.tactile.* (有效性掩码)

    # 5. 生成标准LeRobot列
    # observation.state = 手部关键点(63) + 设备位姿(16)
    observation_state = np.concatenate([
        row_data["status.hands.landmarks_world"],  # 63
        row_data["status.pose.world_device"]       # 16
    ])  # shape: (79,)
    row_data["observation.state"] = observation_state

    # action = [left_total, right_total, left_xyz(3), right_xyz(3)]
    action = np.array([
        row_data["action.tactile.left.force_total"],   # 位置0
        row_data["action.tactile.right.force_total"],  # 位置1
        *row_data["action.tactile.left.force_xyz"],    # 位置2,3,4
        *row_data["action.tactile.right.force_xyz"]    # 位置5,6,7
    ])  # shape: (8,)
    row_data["action"] = action

    return row_data
```

**3. 触觉数据打包（schema.py中的create_tactile_placeholders）**
```python
def create_tactile_placeholders(
    force_left: Optional[Tuple[float, np.ndarray]] = None,
    force_right: Optional[Tuple[float, np.ndarray]] = None
) -> Dict[str, Any]:
    """创建触觉数据列（实际值或NaN占位符）"""

    if force_left and force_right:
        # 解包元组
        left_mag, left_xyz = force_left   # magnitude: float, xyz: array[3]
        right_mag, right_xyz = force_right

        return {
            "action.tactile.left.force_total": np.float32(left_mag),
            "action.tactile.right.force_total": np.float32(right_mag),
            "action.tactile.left.force_xyz": left_xyz.astype(np.float32),   # [fx, fy, fz]
            "action.tactile.right.force_xyz": right_xyz.astype(np.float32),
            "action.mask.tactile.left.force_total": True,
            "action.mask.tactile.right.force_total": True,
            "action.mask.tactile.left.force_xyz": np.array([True, True, True]),
            "action.mask.tactile.right.force_xyz": np.array([True, True, True]),
        }
    else:
        # 无触觉数据时返回NaN
        return {
            "action.tactile.left.force_total": np.float32(np.nan),
            "action.tactile.right.force_total": np.float32(np.nan),
            "action.tactile.left.force_xyz": np.full(3, np.nan, dtype=np.float32),
            "action.tactile.right.force_xyz": np.full(3, np.nan, dtype=np.float32),
            "action.mask.tactile.*": False,  # 掩码全为False
        }
```

**关键实现细节**：
- **时间对齐**：触觉CSV数据通过线性插值对齐到每个相机帧的时间戳
- **xyz分力提取**：从CSV的`_fx`, `_fy`, `_fz`列提取，而非仅使用`_magnitude`
- **数据打包**：`get_force_at_time`返回`((mag1, xyz1), (mag2, xyz2))`元组格式
- **标准列生成**：`observation.state`和`action`遵循LeRobot可视化工具的标准命名
- **掩码机制**：`action.mask.*`列标识哪些数据点有效（True）或缺失（False）

### 架构

```
aria2lerobot/
├── __init__.py                  # 包接口
├── __main__.py                  # CLI入口点
├── convert.py                   # 核心转换逻辑（含触觉支持）
├── schema.py                    # LeRobot模式定义（含触觉）
├── video.py                     # H.264/H.265视频编码
├── io_parquet.py                # 增量parquet写入
├── visualize.py                 # 可视化视频生成（手部+眼动）
├── convert_with_haptics.py      # 触觉数据专用转换工具
├── visualize_with_haptics.py    # 触觉数据专用可视化工具
├── test_haptic_integration.sh   # 触觉集成测试脚本
├── utils.py                     # 日志记录、验证实用程序
├── README.md                    # 本文件
├── README_HAPTIC.md             # 触觉数据集成完整指南
└── QUICK_START.md               # 触觉功能快速开始
```

### 依赖

- **核心**：`pyarrow`、`pandas`、`numpy`
- **视频**：`opencv-python` 或 `imageio-ffmpeg`
- **进度**：`tqdm`
- **序列化**：`orjson`（可选，更快的JSON）
- **Aria访问**：`aria_api`（包含在父目录中）

### 性能优化

- **流式**：从不将完整数据集加载到内存
- **矢量化**：坐标变换的NumPy操作
- **压缩**：parquet文件的Zstandard压缩
- **并行**：视频编码可利用多核
- **缓存**：缓存相机内参以避免重复查询

## 🔮 未来增强

### 计划功能

- [x] **LeRobot v3.0格式**：完全符合官方v3.0规范的输出格式
- [x] **分块架构**：支持chunk-000/结构的可扩展数据集
- [x] **Parquet元数据**：v3.0标准tasks.parquet和episodes/格式
- [x] **episode_chunk列**：v3.0必需的数据分片支持
- [x] **路径模板**：正确的chunk_index和file_index参数格式
- [x] **可视化功能**：手部关键点和眼动数据可视化视频
- [x] **彩虹手指颜色**：不同手指使用不同颜色便于区分
- [x] **FPS修复**：正确的FPS估算和RGB颜色通道处理
- [x] **Project Aria兼容**：符合官方21点手部模型规范
- [x] **畸变校正**：使用Project Aria官方投影函数提高精度
- [x] **可视化旋转**：可选90度图像旋转（仅影响可视化视频）
- [x] **触觉数据支持**：外部触觉传感器CSV集成和可视化
- [x] **lerobot-dataset-viz兼容**：完全支持官方可视化工具
- [x] **多剧集支持**：基于CSV分割点的多episode转换（--split-csv）
- [x] **多episode可视化**：为每个episode生成独立可视化视频（正确的帧和触觉对齐）
- [ ] **点云集成**：可选3D点云数据导出
- [ ] **自定义动作空间**：用户定义的动作列模式
- [ ] **分布式处理**：超大数据集的多进程转换
- [ ] **质量过滤器**：自动过滤低质量帧

### 可扩展性

转换器设计易于扩展：

```python
# 自定义模式扩展
from aria2lerobot.schema import LeRobotSchema, ColumnSpec

custom_schema = LeRobotSchema(
    version="custom_v1.0",
    columns={
        **LEROBOT_V3_SCHEMA.columns,
        "custom.feature": ColumnSpec((128,), "float32", "自定义特征向量"),
    },
    required_columns=LEROBOT_V3_SCHEMA.required_columns + ["custom.feature"],
    video_columns=LEROBOT_V3_SCHEMA.video_columns,
)
```

## 📄 许可证

MIT许可证 - 详见LICENSE文件。

## 🙏 致谢

- **Project Aria**：提供全面的多模态数据集格式
- **LeRobot**：提供标准化的机器人数据集模式
- **aria_api**：提供可靠的底层数据访问和坐标变换

## 📖 相关文档

- [触觉数据集成完整指南](README_HAPTIC.md) - 外部触觉传感器集成详细说明
- [触觉功能快速开始](QUICK_START.md) - 触觉数据转换和可视化快速入门
- [实施总结](IMPLEMENTATION_SUMMARY.md) - 技术实现细节

---

*为机器人学习社区用❤️构建*