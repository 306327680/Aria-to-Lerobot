# Aria2LeRobot 触觉数据集成

支持在Aria数据转换和可视化中集成触觉传感器数据。

## 功能特性

### 1. 数据转换（Parquet格式）
- 将Aria VRS/MPS数据转换为LeRobot格式
- 集成触觉传感器CSV数据
- 自动时间对齐（通过偏移参数）
- 触觉数据在开始前填充为0

### 2. 可视化生成
- 生成带手部关键点的视频
- 添加两个触觉力度指示条
- 最大力度可配置（默认150）
- 实时显示力度数值和刻度

## 文件说明

### 核心脚本

1. **convert_with_haptics.py**
   - 将Aria+触觉数据转换为Parquet格式
   - 支持时间对齐
   - 生成元数据JSON

2. **visualize_with_haptics.py**
   - 生成带触觉指示条的MP4视频
   - 支持力度范围配置
   - 自动颜色编码（传感器1:橙色, 传感器2:绿色）

3. **test_haptic_integration.sh**
   - 完整流程测试脚本
   - 自动检查文件和依赖
   - 一键生成数据集和视频

## 使用方法

### 快速开始

```bash
# 1. 进入目录
cd /Volumes/BU100/Aria_data/aria2lerobot

# 2. 运行测试脚本（使用默认参数）
chmod +x test_haptic_integration.sh
./test_haptic_integration.sh
```

### 自定义参数

#### 生成Parquet数据集

```bash
python3 convert_with_haptics.py \
    /path/to/recording.vrs \
    /path/to/output/dataset \
    --haptic-csv /path/to/haptic.csv \
    --haptic-offset 1.31 \
    --fps 10 \
    --size 224 224
```

**参数说明：**
- `vrs_file`: VRS文件路径
- `output_dir`: 输出目录
- `--haptic-csv`: 触觉CSV文件路径
- `--haptic-offset`: 时间偏移（秒），触觉数据相对于Aria数据需要延后的时间
- `--fps`: 采样率（Hz），默认10
- `--size`: 图像尺寸，默认224x224

#### 生成可视化视频

```bash
python3 visualize_with_haptics.py \
    /path/to/recording.vrs \
    /path/to/output/video.mp4 \
    --haptic-csv /path/to/haptic.csv \
    --haptic-offset 1.31 \
    --max-force 150 \
    --max-frames 300 \
    --stride 1
```

**参数说明：**
- `vrs_file`: VRS文件路径
- `output_video`: 输出视频路径
- `--haptic-csv`: 触觉CSV文件路径
- `--haptic-offset`: 时间偏移（秒）
- `--max-force`: 最大力度值，用于归一化显示，默认150
- `--max-frames`: 最大帧数限制（可选）
- `--stride`: 帧间隔，默认1（处理所有帧）

## 时间对齐

### 确定时间偏移

使用 `aria_haptic_alignment` 库自动计算最优时间偏移：

```bash
cd /Volumes/BU100/Aria_data/aria_haptic_alignment
python3 aria_haptic_align.py
```

输出示例：
```
=== Time Alignment Results ===
Method: xcorr
Optimal offset: 1.310 seconds  # ← 使用此值
Max similarity score: 0.1685
```

将得到的偏移值传递给 `--haptic-offset` 参数。

### 时间对齐原理

- Aria数据时间戳从第一个有效数据开始归零
- 触觉CSV数据使用`relative_time`列（也是从0开始）
- `haptic_offset`表示触觉数据需要向后偏移的秒数
- 在偏移之前的时间点，触觉力度填充为0

## 数据格式

### 输入格式

#### VRS文件
- Project Aria标准VRS格式
- 需要配套MPS产物（自动发现）

#### 触觉CSV文件
期望格式：
```csv
timestamp,relative_time,Sensor1_magnitude,Sensor2_magnitude
1234567890.123,0.0,12.5,8.3
1234567890.223,0.1,15.2,9.1
...
```

必需列：
- `relative_time` 或 `timestamp`: 时间戳
- `*_magnitude`: 力度magnitude列（至少1个）

### 输出格式

#### Parquet数据集

生成文件：
- `data.parquet`: 主数据文件
- `metadata.json`: 元数据信息
- `images/`: 图像文件夹

Parquet列：
| 列名 | 类型 | 说明 |
|-----|------|------|
| frame_id | int64 | 帧ID |
| timestamp_ns | int64 | 时间戳（纳秒） |
| time_s | float64 | 相对时间（秒） |
| image_path | string | 图像路径 |
| right_thumb_index_distance | float32 | 右手拇指-食指距离（米） |
| haptic_force_sensor1 | float32 | 传感器1力度 |
| haptic_force_sensor2 | float32 | 传感器2力度 |

#### 可视化视频

特性：
- 分辨率：与输入VRS相同
- 帧率：10 FPS
- 编码：MP4V
- 包含：
  - 原始RGB图像
  - 两个触觉力度指示条（右侧）
  - 时间戳显示（左上角）
  - 力度刻度和数值

## 示例：测试数据

### 文件位置

```bash
VRS文件:    /Volumes/BU100/Aria_data/glue.vrs
触觉CSV:    /Volumes/BU100/Aria_data/sensor_data/sequence_2_20250927_203243.csv
时间偏移:   1.31秒（通过对齐算法得出）
```

### 运行示例

```bash
# 进入工作目录
cd /Volumes/BU100/Aria_data/aria2lerobot

# 生成数据集
python3 convert_with_haptics.py \
    /Volumes/BU100/Aria_data/glue.vrs \
    ./output/glue_dataset \
    --haptic-csv /Volumes/BU100/Aria_data/sensor_data/sequence_2_20250927_203243.csv \
    --haptic-offset 1.31 \
    --fps 10

# 生成视频
python3 visualize_with_haptics.py \
    /Volumes/BU100/Aria_data/glue.vrs \
    ./output/glue_visualization.mp4 \
    --haptic-csv /Volumes/BU100/Aria_data/sensor_data/sequence_2_20250927_203243.csv \
    --haptic-offset 1.31 \
    --max-force 150
```

## 加载数据

### Python示例

```python
import pandas as pd
import pyarrow.parquet as pq

# 加载Parquet文件
df = pd.read_parquet('output/glue_dataset/data.parquet')

# 查看数据
print(df.head())
print(df.columns)
print(df['haptic_force_sensor1'].describe())

# 筛选有力度的帧
force_frames = df[df['haptic_force_sensor1'] > 0]
print(f"有效力度帧数: {len(force_frames)}")

# 加载元数据
import json
with open('output/glue_dataset/metadata.json') as f:
    metadata = json.load(f)
    print(metadata)
```

## 可视化说明

### 指示条布局

```
┌─────────────────────────────────────┐
│                          [12.5]     │  ← 力度数值
│                          ┌──┐ ┌──┐  │
│                          │▓▓│ │▓▓│  │  ← 力度条
│                          │▓▓│ │░░│  │
│                          │▓▓│ │░░│  │
│                        150─┤  │  │  │  ← 刻度线
│                          │▓▓│ │░░│  │
│                          │▓▓│ │░░│  │
│                          │░░│ │░░│  │
│                          └──┘ └──┘  │
│                         Sensor1     │  ← 传感器标签
│                         Sensor2     │
└─────────────────────────────────────┘
```

### 颜色编码
- **传感器1**: 橙色（RGB: 255, 165, 0）
- **传感器2**: 绿色（RGB: 0, 255, 0）
- **背景框**: 深灰色
- **刻度/边框**: 浅灰色

## 故障排除

### 常见问题

**1. 时间对齐不准确**
- 使用 `aria_haptic_alignment` 重新计算偏移
- 检查CSV的`relative_time`列是否从0开始
- 尝试微调`--haptic-offset`参数

**2. 触觉数据全为0**
- 检查时间偏移是否过大
- 确认CSV文件包含有效的magnitude列
- 验证CSV时间范围与VRS重叠

**3. 视频生成失败**
- 确保安装了OpenCV: `pip install opencv-python`
- 检查磁盘空间
- 减少`--max-frames`或增加`--stride`

**4. Parquet文件损坏**
- 检查pyarrow版本: `pip install --upgrade pyarrow`
- 确保有写入权限
- 尝试更换输出目录

## 依赖项

```bash
pip install numpy pandas pyarrow opencv-python tqdm
```

## 许可证

内部使用

## 作者

Generated with Claude Code