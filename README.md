# Aria to LeRobot/GR00T Converter

将 Project Aria VRS/MPS 数据转换为 LeRobot v3.0 和 NVIDIA GR00T 兼容格式。

## 目录

- [快速开始](#快速开始)
- [完整工作流程](#完整工作流程)
- [数据模态说明](#数据模态说明)
- [脚本说明](#脚本说明)
- [维度提取工具](#维度提取工具)
- [参数说明](#参数说明)
- [故障排除](#故障排除)

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

### 1️⃣ 数据准备

确保你有以下文件：

```
/media/echo/BU100/Aria_data/
├── glue.vrs                    # Aria VRS 录制文件
├── mps_glue_vrs/               # MPS 处理输出
├── sensor_data/
│   └── sequence_2_*.csv        # 触觉传感器数据
└── aria_haptic_alignment/
    └── glue_alignment_segments.csv  # 分段信息
```

### 2️⃣ 转换为 LeRobot/GR00T 格式

使用流水线脚本进行转换：

```bash
bash aria2lerobot/scripts/aria_to_groot_pipeline.sh \
  --vrs glue.vrs \
  --output output_glue \
  --haptic-csv sensor_data/sequence_2_20250927_203243.csv \
  --haptic-offset 1.21 \
  --split-csv aria_haptic_alignment/glue_alignment_segments.csv \
  --task-name "pick and place glue"
```

输出结构：

```
output_glue/
├── data/
│   └── chunk-000/
│       ├── episode_000000.parquet
│       ├── episode_000001.parquet
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

### 3️⃣ (可选) 创建数据集变体用于实验

从完整数据集提取或计算特定特征，创建不同的数据集变体用于对照实验。

#### 可用的数据集变体

| 数据集名称 | State 维度 | Action 维度 | 用途 | 格式 |
|-----------|-----------|------------|------|------|
| `output_glue` | 79 维（手部+姿态） | 8 维（完整力数据） | 完整基线数据集 | GR00T |
| `output_glue_right_hand_force` | 63 维（右手关键点） | 2 维（力的大小） | 简化的手部+力训练 | GR00T |
| `output_glue_pinch_distance` | 1 维（拇指-食指距离） | 2 维（力的大小） | 最小化特征实验 | GR00T |
| `output_glue_pinch_viz` | 7 维（拇指xyz+食指xyz+距离） | 2 维（力的大小） | 捏合动作可视化 | LeRobot |

#### 工具1: extract_dimensions.py - 维度提取

从完整数据集中提取指定维度的子集。

💡 **查看完整帮助**：
```bash
python aria2lerobot/tools/extract_dimensions.py -h
```

**输出格式选项**：
- **`--format groot`** (默认): 用于 GR00T 训练
- **`--format lerobot`**: 用于 LeRobot 可视化

**示例1: 右手关键点 + 力的大小（训练用）**

```bash
python aria2lerobot/tools/extract_dimensions.py \
  output_glue \
  output_glue_right_hand_force \
  --state-dims 0:63 \
  --action-dims 0,1 \
  --state-name "right_hand" \
  --action-name "force_magnitude" \
  --format groot
```

生成数据集：
- State: 63 维（右手21个关键点 × 3坐标）
- Action: 2 维（左右手力的大小）
- 格式：GR00T 训练格式

**示例2: 右手关键点 + 力的大小（可视化用）**

```bash
python aria2lerobot/tools/extract_dimensions.py \
  output_glue \
  output_glue_right_hand_force_viz \
  --state-dims 0:63 \
  --action-dims 0,1 \
  --state-name "right_hand" \
  --action-name "force_magnitude" \
  --format lerobot
```

然后可视化查看：
```bash
conda activate lerobot
lerobot-dataset-viz \
  --repo-id output_glue_right_hand_force_viz \
  --root /media/echo/BU100/Aria_data \
  --episode-index 10
```

#### 工具2: extract_pinch_distance.py - 捏合距离

计算拇指尖和食指尖之间的欧几里得距离作为 1D 状态特征。

**用途**：测试模型能否仅从捏合距离学习到力的控制关系。

```bash
python aria2lerobot/tools/extract_pinch_distance.py \
  output_glue \
  output_glue_pinch_distance \
  --action-dims 0,1 \
  --format groot
```

生成数据集：
- State: 1 维（拇指-食指距离，单位：米）
- Action: 2 维（左右手力的大小）
- 格式：GR00T 训练格式

**距离范围**：约 5.7-9.3 cm（因 episode 而异）

#### 工具3: extract_pinch_features.py - 捏合特征（位置+距离）

提取拇指和食指的3D位置加上它们之间的距离，共7维特征。

**用途**：既包含空间信息又包含距离信息，适合可视化分析捏合动作。

```bash
python aria2lerobot/tools/extract_pinch_features.py \
  output_glue \
  output_glue_pinch_viz \
  --action-dims 0,1 \
  --format lerobot
```

生成数据集：
- State: 7 维 = thumb_xyz (3) + index_xyz (3) + distance (1)
- Action: 2 维（左右手力的大小）
- 格式：LeRobot 可视化格式

**可视化示例**：
```bash
conda activate lerobot
lerobot-dataset-viz \
  --repo-id output_glue_pinch_viz \
  --root /media/echo/BU100/Aria_data \
  --episode-index 26  # Episode 26 有最大的捏合动作变化 (1.2cm)
```

#### 完整数据集生成流程

```bash
# 步骤1: 转换完整基线数据
bash aria2lerobot/scripts/aria_to_groot_pipeline.sh \
  --vrs glue.vrs \
  --output output_glue \
  --haptic-csv sensor_data/sequence_2_20250927_203243.csv \
  --haptic-offset 1.21 \
  --split-csv aria_haptic_alignment/glue_alignment_segments.csv \
  --task-name "pick and place glue"

# 步骤2: 生成右手+力数据集（训练用）
python aria2lerobot/tools/extract_dimensions.py \
  output_glue output_glue_right_hand_force \
  --state-dims 0:63 --action-dims 0,1 \
  --state-name "right_hand" --action-name "force_magnitude" \
  --format groot

# 步骤3: 生成捏合距离数据集（最小化特征）
python aria2lerobot/tools/extract_pinch_distance.py \
  output_glue output_glue_pinch_distance \
  --action-dims 0,1 --format groot

# 步骤4: 生成可视化数据集（检查数据质量）
python aria2lerobot/tools/extract_pinch_features.py \
  output_glue output_glue_pinch_viz \
  --action-dims 0,1 --format lerobot
```

#### 其他维度提取示例

**只要手部 + 所有力数据**

```bash
python aria2lerobot/tools/extract_dimensions.py \
  output_glue output_glue_hand_only \
  --state-dims 0:63 \
  --state-name "right_hand"
```

结果：
- State: 63 维（右手）
- Action: 8 维（全部力数据）

**全部状态 + 只要力的大小（合力）**

```bash
python aria2lerobot/tools/extract_dimensions.py \
  output_glue output_glue_force_only \
  --action-dims 0,1 \
  --action-name "force_magnitude"
```

结果：
- State: 79 维（右手 + gaze + pose）
- Action: 2 维（左右手力的大小）

### 4️⃣ 验证数据集

**验证数据加载**：
```bash
python aria2lerobot/tools/load_groot_dataset.py output_glue --sample-index 10
```

**可视化检查**（LeRobot 格式数据集）：
```bash
conda activate lerobot
lerobot-dataset-viz \
  --repo-id output_glue_pinch_viz \
  --root /media/echo/BU100/Aria_data \
  --episode-index 26
```

💡 **选择有动作的 episode**：不同 episode 的捏合动作变化不同，Episode 26 有最大变化（1.2cm）

### 5️⃣ 在 GR00T 中训练

```bash
cd /path/to/Isaac-GR00T
bash train_aria.sh
```

---

## 数据模态说明

### 完整转换后的数据结构

#### observation.state (79 维)

| 维度范围 | 内容 | 描述 |
|---------|------|------|
| 0:63 | right_hand | 右手21个关键点 (21 × 3 坐标) |
| 63:65 | gaze | 眼睛注视点 UV 归一化坐标 [0-1] |
| 65:79 | device_pose | 设备位姿 (4×4 矩阵的14个元素) |

**注意**: 当前代码存储的是**右手**数据（2025-10-02 更新）

#### action (8 维)

触觉传感器力数据结构：`[left_total, right_total, left_xyz(3), right_xyz(3)]`

**CSV 列到 action 维度的映射**：
- CSV 中的 `Ring Tip_magnitude` 和 `Ring Tip_fx/fy/fz` → left (传感器1)
- CSV 中的 `Little Tip_magnitude` 和 `Little Tip_fx/fy/fz` → right (传感器2)

| 维度 | 内容 | 描述 | CSV 来源 |
|-----|------|------|---------|
| 0 | left.force_total | 左手触觉传感器力的合力大小（标量） | Ring Tip_magnitude |
| 1 | right.force_total | 右手触觉传感器力的合力大小（标量） | Little Tip_magnitude |
| 2 | left.force_xyz[0] | 左手力的 X 分量 | Ring Tip_fx |
| 3 | left.force_xyz[1] | 左手力的 Y 分量 | Ring Tip_fy |
| 4 | left.force_xyz[2] | 左手力的 Z 分量 | Ring Tip_fz |
| 5 | right.force_xyz[0] | 右手力的 X 分量 | Little Tip_fx |
| 6 | right.force_xyz[1] | 右手力的 Y 分量 | Little Tip_fy |
| 7 | right.force_xyz[2] | 右手力的 Z 分量 | Little Tip_fz |

**常用维度组合**：
- 只要力的大小（合力）：`--action-dims 0,1`
- 只要左手完整数据：`--action-dims 0,2,3,4`（大小 + XYZ）
- 只要右手完整数据：`--action-dims 1,5,6,7`（大小 + XYZ）
- 只要 XYZ 分量：`--action-dims 2,3,4,5,6,7`

#### video

- **observation.images.main**: 1408×1408×3 RGB 视频帧（来自 Aria RGB 相机）

### MediaPipe Hands 关键点索引

右手21个关键点的索引（每个点3个坐标 x,y,z）：

```
关键点编号 (索引 = 编号 × 3)
0:  WRIST (手腕)
1:  THUMB_CMC
2:  THUMB_MCP
3:  THUMB_IP
4:  THUMB_TIP (大拇指尖)
5:  INDEX_FINGER_MCP
6:  INDEX_FINGER_PIP
7:  INDEX_FINGER_DIP
8:  INDEX_FINGER_TIP (食指尖)
9:  MIDDLE_FINGER_MCP
10: MIDDLE_FINGER_PIP
11: MIDDLE_FINGER_DIP
12: MIDDLE_FINGER_TIP (中指尖)
13: RING_FINGER_MCP
14: RING_FINGER_PIP
15: RING_FINGER_DIP
16: RING_FINGER_TIP (无名指尖)
17: PINKY_MCP
18: PINKY_PIP
19: PINKY_DIP
20: PINKY_TIP (小指尖)
```

**示例: 提取特定关键点**

只要大拇指尖和食指尖（捏取动作）：
- THUMB_TIP: 索引 4 → 维度 12, 13, 14
- INDEX_FINGER_TIP: 索引 8 → 维度 24, 25, 26

```bash
python aria2lerobot/tools/extract_dimensions.py \
  output_glue \
  output_glue_pinch \
  --state-dims 12,13,14,24,25,26 \
  --action-dims 0,1 \
  --state-name "pinch_fingertips" \
  --action-name "force_magnitude"
```

---

## 脚本说明

### aria_to_groot_pipeline.sh

单任务转换流水线，包含完整的转换、验证和测试步骤。

**主要步骤**:
1. 激活 conda 环境
2. 转换 VRS/MPS → LeRobot v3.0 格式
3. 生成 GR00T 兼容的 metadata (modality.json, *.jsonl)
4. 重新计算统计信息 (stats.json)
5. 验证 GR00T 格式
6. (可选) 测试数据加载

**使用方法**:

```bash
bash aria2lerobot/scripts/aria_to_groot_pipeline.sh \
  --vrs <vrs_file> \
  --output <output_dir> \
  --haptic-csv <haptic_csv> \
  --haptic-offset <offset> \
  --split-csv <split_csv> \
  --task-name <task_name> \
  [--test-loading]
```

**参数**:
- `--vrs`: VRS 文件路径
- `--output`: 输出目录
- `--haptic-csv`: 触觉传感器 CSV 文件
- `--haptic-offset`: 触觉数据时间偏移（秒）
- `--split-csv`: 分段 CSV 文件
- `--task-name`: 任务名称
- `--test-loading`: (可选) 测试数据加载

### batch_aria_to_groot.sh

批量处理多个任务的自动化脚本。

**配置任务列表**:

编辑脚本中的 `TASKS` 数组：

```bash
declare -a TASKS=(
    "glue.vrs|output_glue|sensor_data/sequence_2_*.csv|1.21|aria_haptic_alignment/glue_alignment_segments.csv|pick and place glue|Assembly task using glue"
    "pen.vrs|output_pen|sensor_data/sequence_3_*.csv|1.13|aria_haptic_alignment/pen_alignment_segments.csv|pick and place pen|Assembly task using pen"
)
```

**使用方法**:

```bash
# 处理所有任务
bash aria2lerobot/scripts/batch_aria_to_groot.sh

# 只处理特定任务（跳过已存在的）
bash aria2lerobot/scripts/batch_aria_to_groot.sh

# 强制重新处理所有任务
# (需要手动编辑脚本，设置 FORCE_REPROCESS=true)
```

---

## 维度提取工具

### extract_dimensions.py

从完整数据集中提取指定维度，用于创建多个实验对照组。

**💡 查看完整帮助**: `python aria2lerobot/tools/extract_dimensions.py -h`

**语法**:

```bash
python aria2lerobot/tools/extract_dimensions.py \
  <source_dataset> \
  <output_dataset> \
  [--state-dims SPEC] \
  [--action-dims SPEC] \
  [--state-name NAME] \
  [--action-name NAME] \
  [--format {groot,lerobot}]
```

**参数说明**:

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `source_dataset` | 必需 | 源数据集目录 | - |
| `output_dataset` | 必需 | 输出数据集目录 | - |
| `--state-dims` | 可选 | State 维度（如 `0:63`, `3,7,12`） | 保留全部 |
| `--action-dims` | 可选 | Action 维度（如 `0,1`, `2:5`） | 保留全部 |
| `--state-name` | 可选 | State 模态名称 | `extracted_state` |
| `--action-name` | 可选 | Action 模态名称 | `extracted_action` |
| `--format` | 可选 | 输出格式：`groot` (训练) 或 `lerobot` (可视化) | `groot` |

**维度指定格式**:

- 单个维度: `3` 或 `7`
- 多个维度: `3,7,10`
- 范围: `0:63` (从0到62，含)
- 混合: `0,5,10:15,20`

**Action 维度映射** (重要！):

| 维度 | 含义 | CSV 来源 |
|------|------|---------|
| 0 | 左手力的大小（合力） | Ring Tip_magnitude |
| 1 | 右手力的大小（合力） | Little Tip_magnitude |
| 2-4 | 左手力的 XYZ 分量 | Ring Tip_fx/fy/fz |
| 5-7 | 右手力的 XYZ 分量 | Little Tip_fx/fy/fz |

**常用配置示例**:

#### 1. 右手 + 力的大小（训练用）

```bash
python aria2lerobot/tools/extract_dimensions.py \
  output_glue output_glue_train \
  --state-dims 0:63 \
  --action-dims 0,1 \
  --state-name "right_hand" \
  --action-name "force_magnitude" \
  --format groot
```

#### 2. 右手 + 力的大小（可视化用）

```bash
python aria2lerobot/tools/extract_dimensions.py \
  output_glue output_glue_viz \
  --state-dims 0:63 \
  --action-dims 0,1 \
  --state-name "right_hand" \
  --action-name "force_magnitude" \
  --format lerobot
```

#### 3. 只有手部关键点 + 全部力数据

```bash
python aria2lerobot/tools/extract_dimensions.py \
  output_glue output_glue_hand_only \
  --state-dims 0:63 \
  --state-name "right_hand"
```

#### 4. 全部状态 + 只要力的大小

```bash
python aria2lerobot/tools/extract_dimensions.py \
  output_glue output_glue_force_only \
  --action-dims 0,1 \
  --action-name "force_magnitude"
```

#### 5. 捏取动作（拇指+食指尖）

```bash
python aria2lerobot/tools/extract_dimensions.py \
  output_glue output_glue_pinch \
  --state-dims 12,13,14,24,25,26 \
  --action-dims 0,1 \
  --state-name "pinch_fingertips" \
  --action-name "force_magnitude"
```

#### 6. 所有指尖（5个指尖）

```bash
python aria2lerobot/tools/extract_dimensions.py \
  output_glue output_glue_fingertips \
  --state-dims 12,13,14,24,25,26,36,37,38,48,49,50,60,61,62 \
  --action-dims 0,1 \
  --state-name "five_fingertips" \
  --action-name "force_magnitude"
```

---

## 参数说明

### 转换参数 (aria2lerobot/__main__.py)

#### 必需参数

- `vrs_path`: VRS 文件路径
- `output_dir`: 输出目录

#### 可选参数

**数据源**:
- `--mps-dir`: MPS 目录路径（默认自动检测）
- `--fps`: 目标 FPS（默认自动检测）
- `--max-slop-ns`: 时间对齐容差（纳秒，默认5ms）

**格式选项**:
- `--legacy-v21`: 使用 LeRobot v2.1 格式
- `--groot-format`: 生成 GR00T 兼容的 modality.json

**处理选项**:
- `--stride`: 帧间隔（1=每帧，2=每隔一帧）
- `--max-frames`: 最大帧数
- `--overwrite`: 覆盖已存在的输出目录

**可视化**:
- `--create-visualization`: 创建可视化视频
- `--rotate-visualization`: 将视频旋转90度

**触觉数据**:
- `--haptic-csv`: 触觉传感器 CSV 文件
- `--haptic-offset`: 时间偏移（秒）

**分段**:
- `--split-csv`: 分段 CSV 文件

**任务元数据**:
- `--task-name`: 任务名称（默认: "aria_recording"）
- `--task-description`: 任务描述

**日志**:
- `--log-level`: 日志级别 (DEBUG, INFO, WARNING, ERROR)
- `--quiet`: 静默模式

### 维度提取参数 (tools/extract_dimensions.py)

#### 必需参数

- `source_dataset`: 源数据集目录
- `output_dataset`: 输出数据集目录

#### 可选参数

- `--state-dims`: 要提取的 state 维度
- `--action-dims`: 要提取的 action 维度
- `--state-name`: state 模态名称（默认: "extracted_state"）
- `--action-name`: action 模态名称（默认: "extracted_action"）

**注意**: `--state-dims` 和 `--action-dims` 至少要指定一个

---

## 转换示例命令

### 示例1: 基础转换（完整数据）

```bash
bash aria2lerobot/scripts/aria_to_groot_pipeline.sh \
  --vrs glue.vrs \
  --output output_glue \
  --haptic-csv sensor_data/sequence_2_20250927_203243.csv \
  --haptic-offset 1.21 \
  --split-csv aria_haptic_alignment/glue_alignment_segments.csv \
  --task-name "pick and place glue"
```

输出:
- State: 79 维（右手63 + gaze2 + pose14）
- Action: 8 维（全部力数据）

### 示例2: 转换 + 测试加载

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

### 示例3: 完整流程（转换 → 提取多个实验配置）

```bash
# 步骤1: 转换完整数据
bash aria2lerobot/scripts/aria_to_groot_pipeline.sh \
  --vrs glue.vrs \
  --output output_glue \
  --haptic-csv sensor_data/sequence_2_20250927_203243.csv \
  --haptic-offset 1.21 \
  --split-csv aria_haptic_alignment/glue_alignment_segments.csv \
  --task-name "pick and place glue"

# 步骤2: 创建对照组1 - 只有右手 + 力大小
python aria2lerobot/tools/extract_dimensions.py \
  output_glue output_glue_exp1_hand_force \
  --state-dims 0:63 \
  --action-dims 3,7 \
  --state-name "right_hand" \
  --action-name "force_magnitude"

# 步骤3: 创建对照组2 - 只有指尖 + 力大小
python aria2lerobot/tools/extract_dimensions.py \
  output_glue output_glue_exp2_fingertips_force \
  --state-dims 12,13,14,24,25,26,36,37,38,48,49,50,60,61,62 \
  --action-dims 3,7 \
  --state-name "five_fingertips" \
  --action-name "force_magnitude"

# 步骤4: 创建对照组3 - 手部 + gaze + 力大小
python aria2lerobot/tools/extract_dimensions.py \
  output_glue output_glue_exp3_hand_gaze_force \
  --state-dims 0:65 \
  --action-dims 3,7 \
  --state-name "hand_and_gaze" \
  --action-name "force_magnitude"
```

### 示例4: 批量转换所有任务

```bash
bash aria2lerobot/scripts/batch_aria_to_groot.sh
```

这会自动处理配置文件中的所有任务（glue, pen, screw 等）

---

## 故障排除

### 问题1: KeyError: 'chunk_index' 或 'episode_chunk'

**原因**: info.json 中的路径模式使用了错误的变量名

**解决方案**: 确保 `aria2lerobot/schema.py` 中使用 GR00T 兼容的变量名：

```python
"data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
```

### 问题2: KeyError: 'observation.state' in stats.json

**原因**: stats.json 缺少 q01 和 q99 百分位数

**解决方案**: 重新计算统计信息：

```bash
python aria2lerobot/tools/compute_stats.py output_glue
```

### 问题3: 数据集加载时出现 Segmentation fault

**原因**:
- 内存不足
- 数据维度过大
- 视频解码问题

**解决方案**:
1. 使用维度提取工具减少数据维度
2. 减少 batch size
3. 使用 `--video-backend pyav` 代替 `decord`

### 问题4: 想要左手数据而不是右手

**解决方案**: 修改 `aria2lerobot/schema.py` 中的 `flatten_hand_landmarks` 函数：

```python
# 改为存储左手
if left_landmarks is not None and left_landmarks.shape == (21, 3):
    flattened[:63] = left_landmarks.flatten()
    valid_left = not np.any(np.isnan(left_landmarks))
```

然后重新转换数据。

### 问题5: 触觉数据时间不同步

**原因**: 触觉传感器和 Aria 时钟不同步

**解决方案**: 调整 `--haptic-offset` 参数，使用 `aria_haptic_alignment` 工具确定正确的偏移量。

---

## 相关工具

- **compute_stats.py**: 重新计算数据集统计信息
- **load_groot_dataset.py**: 测试数据集加载
- **convert_to_groot_format.py**: 将现有 LeRobot 数据集转换为 GR00T 格式
- **extract_dimensions.py**: 提取指定维度创建子数据集
- **extract_pinch_distance.py**: 提取捏合距离（1D state）
- **extract_pinch_features.py**: 提取捏合特征（7D state: 位置+距离）

---

## 参考文档

- [LeRobot v3.0 Format](https://github.com/huggingface/lerobot)
- [NVIDIA GR00T](https://developer.nvidia.com/isaac-lab)
- [Project Aria](https://projectaria.com/)
- [MediaPipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker)

---

## 更新日志

### 2025-10-03
- ✅ 创建 4 种数据集变体用于对照实验
- ✅ 添加捏合距离提取工具（1D state）
- ✅ 添加捏合特征提取工具（7D state）
- ✅ 添加 `--format` 参数支持 GR00T/LeRobot 格式切换
- ✅ 更新 README 包含完整数据集生成流程

### 2025-10-02
- ✅ 修改代码存储**右手**而不是左手
- ✅ 修复 info.json 路径模式为 GR00T 兼容格式
- ✅ 添加 q01 和 q99 百分位数到 stats.json
- ✅ 创建维度提取工具用于实验对照组
- ✅ 优化 VRS 文件加载（单次加载用于所有可视化）

### 2025-10-01
- ✅ 添加任务名称和描述参数
- ✅ 创建 GR00T 训练流程
- ✅ 添加批量处理脚本

---

**作者**: Claude Code
**许可证**: MIT
