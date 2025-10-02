# Aria2LeRobot - Project Aria 到 NVIDIA Isaac GR00T 数据转换工具

将 Project Aria VRS 数据转换为 LeRobot v3.0 和 NVIDIA Isaac GR00T 兼容的机器人学习数据集。

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![LeRobot](https://img.shields.io/badge/LeRobot-v3.0-orange.svg)](https://huggingface.co/docs/lerobot)
[![GR00T](https://img.shields.io/badge/NVIDIA-GR00T-76B900.svg)](https://github.com/NVIDIA/Isaac-GR00T)

---

## 📋 目录

- [快速开始](#快速开始)
- [完整工作流程](#完整工作流程)
- [文件结构](#文件结构)
- [脚本说明](#脚本说明)
- [数据集结构](#数据集结构)
- [自定义模态选择](#自定义模态选择)
- [训练配置示例](#训练配置示例)
- [故障排除](#故障排除)

---

## 🚀 快速开始

### 方式 1: 批量处理（推荐）

一次性处理所有 VRS 文件（glue, pen, screw）：

```bash
cd /media/echo/BU100/Aria_data
bash aria2lerobot/scripts/batch_aria_to_groot.sh

# 添加数据集加载测试
bash aria2lerobot/scripts/batch_aria_to_groot.sh --test-loading

# 只处理特定任务
bash aria2lerobot/scripts/batch_aria_to_groot.sh --only glue --only pen

# 并行处理（更快但调试困难）
bash aria2lerobot/scripts/batch_aria_to_groot.sh --parallel
```

### 方式 2: 单个任务处理

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

---

## 📖 完整工作流程

### 步骤 1: 批量转换 VRS 到 GR00T 格式

```bash
bash aria2lerobot/scripts/batch_aria_to_groot.sh
```

**自动完成的操作：**
- ✅ 转换 VRS/MPS 数据到 LeRobot v3.0 格式
- ✅ 集成触觉传感器数据
- ✅ 根据 CSV 分割成多个 episodes
- ✅ 生成 GR00T 兼容的元数据文件
- ✅ 创建可视化视频（每个 episode 一个）
- ✅ 验证 GR00T 兼容性

### 步骤 2: 验证数据集

```bash
# 测试加载转换后的数据集
python3 aria2lerobot/tools/load_groot_dataset.py output_glue --sample-index 100
```

### 步骤 3: 选择训练配置

```bash
# 查看预定义配置
python3 aria2lerobot/tools/aria_groot_config.py
```

**可用配置：**
- **PINCH_GRASP_CONFIG**: 捏取抓握（食指+大拇指，6维 state + 2维 action）
- **FULL_HAND_CONFIG**: 完整手部数据（79维 state + 8维 action）
- **FINGERTIPS_CONFIG**: 手腕+五指尖+注视点（20维 state + 2维 action）

### 步骤 4: 训练模型

```bash
# 运行训练示例
python3 aria2lerobot/tools/train_example_pinch_grasp.py
```

---

## 📁 文件结构

```
aria2lerobot/
├── __init__.py
├── __main__.py              # CLI 入口点
├── convert.py               # 主转换逻辑
├── schema.py                # LeRobot v3.0 schema 定义
├── metadata.py              # 元数据生成（含 GR00T 支持）
├── frame_processing.py      # 帧数据处理
├── chunking.py              # 多 episode 处理
├── visualization.py         # 可视化生成
├── scripts/                 # Shell 脚本
│   ├── batch_aria_to_groot.sh      # 批量处理脚本（推荐）
│   └── aria_to_groot_pipeline.sh   # 单任务处理脚本
├── tools/                   # GR00T 工具
│   ├── aria_groot_config.py        # 训练配置管理
│   ├── convert_to_groot_format.py  # 格式转换工具
│   ├── load_groot_dataset.py       # 数据集加载器
│   └── train_example_pinch_grasp.py # 训练示例
└── README.md                # 本文档
```

---

## 📚 脚本说明

### 转换脚本

| 脚本 | 用途 | 特点 |
|------|------|------|
| `scripts/batch_aria_to_groot.sh` ⭐ | 批量处理所有 VRS 文件 | 自动配置、并行支持、选择性处理 |
| `scripts/aria_to_groot_pipeline.sh` | 单个任务的完整流程 | 灵活参数、自动验证、可选测试 |

### 工具脚本

| 脚本 | 用途 |
|------|------|
| `tools/convert_to_groot_format.py` | 转换现有数据集到 GR00T 格式 |
| `tools/load_groot_dataset.py` | 加载和验证 GR00T 数据集 |
| `tools/aria_groot_config.py` | 训练配置管理工具 |
| `tools/train_example_pinch_grasp.py` | 训练示例代码 |

### 批量脚本选项

```bash
bash aria2lerobot/scripts/batch_aria_to_groot.sh [OPTIONS]

Options:
    --skip-conversion       跳过 VRS 转换（使用已有数据）
    --skip-visualization    跳过可视化视频生成
    --test-loading          转换后测试数据集加载
    --parallel              并行处理（实验性）
    --only TASK_NAME        只处理特定任务（可多次使用）
                           有效值: glue, pen, screw
    -h, --help             显示帮助信息

Examples:
    # 处理所有任务
    bash aria2lerobot/scripts/batch_aria_to_groot.sh

    # 只处理 glue 和 pen
    bash aria2lerobot/scripts/batch_aria_to_groot.sh --only glue --only pen

    # 跳过转换，只更新 GR00T 元数据
    bash aria2lerobot/scripts/batch_aria_to_groot.sh --skip-conversion
```

---

## 📁 数据集结构

转换后的数据集包含：

```
output_glue/
├── meta/
│   ├── modality.json          # ✨ GR00T 模态定义
│   ├── episodes.jsonl         # ✨ GR00T episode 元数据
│   ├── tasks.jsonl            # ✨ GR00T task 元数据
│   ├── info.json              # LeRobot 数据集信息
│   ├── stats.json             # 统计信息
│   ├── tasks.parquet          # LeRobot v3.0 任务（向后兼容）
│   └── episodes/              # LeRobot v3.0 episodes（向后兼容）
│       └── chunk-000/file-000.parquet
├── data/                      # Parquet 数据文件
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
│       └── ...
├── videos/                    # MP4 视频文件
│   └── chunk-000/
│       └── observation.images.main/
│           ├── episode_000000.mp4
│           └── ...
└── visualization_episode_*.mp4  # 可视化视频
```

### 数据模态说明

#### State (observation.state, 79 维)

| 模态 | 索引 | 维度 | 描述 |
|------|------|------|------|
| `hand_landmarks` | 0:63 | 63 | 手部21个关节 × 3坐标 (x,y,z) |
| `gaze` | 63:65 | 2 | 归一化 UV 注视点 [0-1] |
| `device_pose` | 65:79 | 14 | 4×4 变换矩阵的14个元素 |

**注意：** `hand_landmarks` 在手部未检测到时可能为 NaN。

#### Action (action, 8 维)

| 模态 | 索引 | 维度 | 描述 |
|------|------|------|------|
| `ring_tip_force_xyz` | 0:3 | 3 | 无名指尖力向量 (fx, fy, fz) |
| `ring_tip_force_magnitude` | 3:4 | 1 | 无名指尖力大小 |
| `little_tip_force_xyz` | 4:7 | 3 | 小指尖力向量 (fx, fy, fz) |
| `little_tip_force_magnitude` | 7:8 | 1 | 小指尖力大小 |

---

## 🎯 自定义模态选择

### 如何选择特定维度进行训练？

使用 `aria_groot_config.py` 中的配置系统选择特定维度。

### 示例 1: 只使用食指和大拇指作为 State

```python
import sys
sys.path.insert(0, 'aria2lerobot/tools')

from aria_groot_config import PINCH_GRASP_CONFIG, get_state_indices, get_action_indices
from gr00t.data.dataset import LeRobotSingleDataset, ModalityConfig

# 使用预定义配置
config = PINCH_GRASP_CONFIG
state_indices = get_state_indices(config["state"])  # [12, 13, 14, 24, 25, 26]
action_indices = get_action_indices(config["action"])  # [3, 7]

# 加载数据集
modality_configs = {
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["video.main"]
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["state.hand_landmarks"]  # 加载完整的，后续筛选
    ),
    "action": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "action.ring_tip_force_magnitude",
            "action.little_tip_force_magnitude"
        ]
    ),
}

dataset = LeRobotSingleDataset(
    dataset_path="output_glue",
    modality_configs=modality_configs,
    embodiment_tag="new_embodiment",
    video_backend="decord"
)

# 在训练循环中选择维度
import numpy as np
sample = dataset[0]
state = sample["state.hand_landmarks"][:, state_indices]  # (1, 6)
action = np.concatenate([
    sample["action.ring_tip_force_magnitude"],
    sample["action.little_tip_force_magnitude"]
], axis=-1)  # (1, 2)
```

### 示例 2: 自定义配置

```python
# 自定义：只用手腕和食指尖
custom_config = {
    "state": {
        "landmarks": ["WRIST", "INDEX_FINGER_TIP"],  # 6 dims
        "use_gaze": True,  # +2 dims
        "use_pose": False,
    },
    "action": {
        "modalities": ["ring_tip_force_magnitude"],  # 1 dim
    }
}

state_indices = get_state_indices(custom_config["state"])  # [0,1,2, 24,25,26, 63,64]
action_indices = get_action_indices(custom_config["action"])  # [3]
```

### 预定义配置详情

查看所有配置：

```bash
python3 aria2lerobot/tools/aria_groot_config.py
```

输出示例：

```
============================================================
Configuration: PINCH_GRASP_CONFIG
============================================================

State dimensions: 6
  Indices: [12, 13, 14, 24, 25, 26]
  Landmarks: THUMB_TIP, INDEX_FINGER_TIP
  Gaze: No
  Pose: No

Action dimensions: 2
  Indices: [3, 7]
  Modalities: ring_tip_force_magnitude, little_tip_force_magnitude
```

---

## 🔧 训练配置示例

### 完整训练脚本模板

```python
import sys
sys.path.insert(0, '/home/echo/Isaac-GR00T')
sys.path.insert(0, '/media/echo/BU100/Aria_data/aria2lerobot/tools')

from pathlib import Path
import numpy as np
from gr00t.data.dataset import LeRobotSingleDataset, ModalityConfig
from aria_groot_config import PINCH_GRASP_CONFIG, get_state_indices, get_action_indices

# 1. 选择配置
config = PINCH_GRASP_CONFIG
state_indices = get_state_indices(config["state"])
action_indices = get_action_indices(config["action"])

print(f"State dims: {len(state_indices)}, Action dims: {len(action_indices)}")

# 2. 加载数据集
modality_configs = {
    "video": ModalityConfig(
        delta_indices=[0, -1, -2],  # 当前帧 + 前两帧
        modality_keys=["video.main"]
    ),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["state.hand_landmarks"]
    ),
    "action": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "action.ring_tip_force_magnitude",
            "action.little_tip_force_magnitude"
        ]
    ),
}

dataset = LeRobotSingleDataset(
    dataset_path="output_glue",
    modality_configs=modality_configs,
    embodiment_tag="new_embodiment",
    video_backend="decord"
)

# 3. 训练循环
for idx in range(len(dataset)):
    sample = dataset[idx]

    # 提取数据
    video = sample["video.main"]  # (T, H, W, 3)
    state = sample["state.hand_landmarks"][:, state_indices]  # (1, 6)
    action = np.concatenate([
        sample["action.ring_tip_force_magnitude"],
        sample["action.little_tip_force_magnitude"]
    ], axis=-1)  # (1, 2)

    # 你的训练代码
    # loss = model(video, state, action)
    # ...
```

---

## ❓ 故障排除

### 问题 1: `ModuleNotFoundError: No module named 'tyro'`

**解决方案：** 使用 lerobot 环境

```bash
conda activate lerobot
python3 your_script.py
```

### 问题 2: `ValueError: 'aria' is not a valid EmbodimentTag`

**解决方案：** 使用 `"new_embodiment"` 作为 embodiment tag

```python
dataset = LeRobotSingleDataset(
    embodiment_tag="new_embodiment",  # ✓ 正确
    # embodiment_tag="aria",  # ✗ 错误
    ...
)
```

### 问题 3: 缺少 `modality.json` 或 `episodes.jsonl`

**解决方案：** 重新转换时使用批量脚本

```bash
# 方法 1: 使用批量脚本
bash aria2lerobot/scripts/batch_aria_to_groot.sh

# 方法 2: 转换已有数据集
python3 aria2lerobot/tools/convert_to_groot_format.py output_glue
```

### 问题 4: 如何验证数据集是否正确？

```bash
# 方法 1: 使用验证脚本
python3 aria2lerobot/tools/convert_to_groot_format.py output_glue

# 方法 2: 测试加载
python3 aria2lerobot/tools/load_groot_dataset.py output_glue --sample-index 0

# 方法 3: 检查文件
ls -lh output_glue/meta/
# 应该看到: modality.json, episodes.jsonl, tasks.jsonl, info.json, stats.json
```

### 问题 5: observation.state 中有 NaN

这是正常现象！当手部未被检测到时，`hand_landmarks` 会包含 NaN 值。在训练时需要处理：

```python
# 选项 1: 跳过包含 NaN 的样本
if not np.isnan(state).any():
    # 使用这个样本

# 选项 2: 用前一帧填充
state = np.nan_to_num(state, nan=0.0)

# 选项 3: 插值填充
# 使用 pandas 的 interpolate 功能
```

---

## 📊 完整流程时间线

```
1. 批量转换（约 5-15 分钟/任务）
   └─> bash aria2lerobot/scripts/batch_aria_to_groot.sh

2. 验证数据集（约 10 秒）
   └─> python3 aria2lerobot/tools/load_groot_dataset.py output_glue --sample-index 0

3. 查看配置选项（约 1 秒）
   └─> python3 aria2lerobot/tools/aria_groot_config.py

4. 开始训练
   └─> python3 aria2lerobot/tools/train_example_pinch_grasp.py
```

---

## 🎉 完整示例流程

### 场景：训练一个抓取控制器（只用食指+大拇指）

```bash
# 步骤 1: 转换数据
bash aria2lerobot/scripts/batch_aria_to_groot.sh --only glue --test-loading

# 步骤 2: 验证配置
python3 aria2lerobot/tools/aria_groot_config.py

# 步骤 3: 训练
python3 aria2lerobot/tools/train_example_pinch_grasp.py
```

**输出：**
- ✅ 数据集：`output_glue/`（GR00T 兼容）
- ✅ State: 6 维（食指尖 xyz + 大拇指尖 xyz）
- ✅ Action: 2 维（两个传感器的合力）
- ✅ 可视化：每个 episode 的视频

---

## 📚 相关文档

- [NVIDIA Isaac GR00T Documentation](https://github.com/NVIDIA/Isaac-GR00T)
- [LeRobot v3.0 Dataset Format](https://huggingface.co/docs/lerobot/main/lerobot-dataset-v3)
- [Project Aria Documentation](https://facebookresearch.github.io/projectaria_tools/)

---

## 🆘 获取帮助

每个脚本都支持 `--help` 参数：

```bash
bash aria2lerobot/scripts/batch_aria_to_groot.sh --help
bash aria2lerobot/scripts/aria_to_groot_pipeline.sh --help
python3 aria2lerobot/tools/load_groot_dataset.py --help
```

如有问题或改进建议，请提 Issue 或 PR！
