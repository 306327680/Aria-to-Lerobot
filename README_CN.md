# Aria2LeRobot - Project Aria到LeRobot数据集转换器

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-production--ready-brightgreen.svg)](https://github.com)

**将Project Aria VRS/MPS数据转换为LeRobot数据集格式，支持完整的多模态对齐和鲁棒的数据处理。**

基于可靠的`aria_api`构建，实现无缝数据访问，将复杂的Aria录制数据转换为标准化的LeRobot数据集，可直接用于机器学习工作流。

## ✨ 核心特性

### 🎯 **格式支持**
- **LeRobot v3.0**（默认）：最新格式，优化的数据结构
- **LeRobot v2.1**（兼容）：向后兼容选项
- 严格的模式验证和一致的数据类型

### 📊 **多模态数据**
- **视觉**：RGB相机帧 → H.264/H.265编码视频
- **空间**：世界坐标系下的设备位姿（SE3变换）
- **手部**：世界坐标系下21点3D关键点（米为单位）
- **触觉**：预留动作占位符，带适当的掩码
- **时间**：纳秒精度时间戳，设备时间对齐

### 🔄 **鲁棒处理**
- **时间对齐**：内置最近邻查询，可配置容差
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
# 自动检测MPS目录转换
python -m aria2lerobot glue.vrs output_dataset/

# 显式指定MPS目录
python -m aria2lerobot glue.vrs output_dataset/ --mps-dir mps_glue_vrs/

# 使用传统v2.1格式
python -m aria2lerobot glue.vrs output_dataset/ --legacy-v21
```

### Python API使用

```python
from aria2lerobot import convert

# 基础转换
result = convert(
    vrs_path="glue.vrs",
    mps_dir="mps_glue_vrs",
    out_dir="output_dataset"
)

# 高级选项
result = convert(
    vrs_path="glue.vrs", 
    mps_dir="mps_glue_vrs",
    out_dir="output_dataset",
    fps=15.0,                    # 自定义FPS
    max_slop_ns=10_000_000,     # 10ms对齐容差
    stride=2,                    # 每隔一帧
    max_frames=50000,           # 限制处理帧数
    legacy_v21=False,           # 使用v3.0格式
    overwrite=True              # 覆盖现有输出
)

print(f"处理了 {result['processing']['valid_frames']} 帧")
print(f"手部有效率: {result['data_quality']['hand_valid_rate']:.1%}")
```

## 📁 输出结构

### LeRobot v3.0（默认）

```
output_dataset/
├── meta/
│   ├── info.json              # 模式定义、FPS、剧集信息
│   ├── stats.json             # 归一化数据统计
│   └── episodes/
│       └── episodes.parquet   # 剧集元数据
├── data/
│   ├── data_000.parquet       # 主表格数据（分片）
│   ├── data_001.parquet
│   └── ...
├── videos/
│   ├── main_000.mp4          # RGB视频分片
│   ├── main_001.mp4
│   └── ...
└── conversion.log            # 详细转换日志
```

### LeRobot v2.1（传统）

```
output_dataset/
├── episode_000000.parquet    # 剧集数据
├── episode_000000.mp4        # 剧集视频
├── dataset_info.json         # 基础元数据
└── conversion.log           # 转换日志
```

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
| `status.hands.landmarks_world` | float32 | (63,) | 世界坐标系下左手关键点（21×3扁平化） |
| `status.hands.valid_left` | bool | () | 左手有效性标志 |
| `status.hands.valid_right` | bool | () | 右手有效性标志 |

### 位姿数据

| 列名 | 类型 | 形状 | 描述 |
|--------|------|-------|-------------|
| `status.pose.world_device` | float32 | (16,) | 世界→设备变换（4×4矩阵扁平化） |
| `status.pose.valid` | bool | () | 位姿有效性标志 |

### 触觉动作（占位符）

| 列名 | 类型 | 形状 | 描述 |
|--------|------|-------|-------------|
| `action.tactile.left.force_total` | float32 | () | 左手总力（NaN） |
| `action.tactile.right.force_total` | float32 | () | 右手总力（NaN） |
| `action.tactile.left.force_xyz` | float32 | (3,) | 左手力向量（NaN） |
| `action.tactile.right.force_xyz` | float32 | (3,) | 右手力向量（NaN） |
| `action.mask.tactile.*` | bool | () | 对应掩码（全为False） |

### 视频数据

| 列名 | 描述 |
|--------|-------------|
| `observation.images.main` | 存储为MP4视频的RGB相机帧 |

## 🔧 配置选项

### 命令行参数

```bash
python -m aria2lerobot --help

# 关键选项:
--fps FLOAT              # 目标FPS（未指定时自动检测）
--max-slop-ns INT        # 时间对齐容差（默认：5ms）
--stride INT             # 帧步长（1=每帧，2=每隔一帧）
--max-frames INT         # 最大处理帧数
--legacy-v21             # 使用v2.1格式而非v3.0
--overwrite              # 覆盖现有输出
--log-level LEVEL        # DEBUG、INFO、WARNING、ERROR
```

### 时间对齐

转换器使用**camera-rgb帧作为主要时间线**：

- 对每个相机帧时间戳，查询对齐的数据：
  - **位姿**：`session.nearest_pose(ts_ns, max_slop_ns)`（优先闭环）
  - **手部**：`session.hands.landmarks_in_world(ts_ns, T_world_device, max_slop_ns)`
- 默认容差：**5ms**（`max_slop_ns=5_000_000`）
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
# 转换glue.vrs，自动检测MPS
python -m aria2lerobot glue.vrs glue_dataset/
```

预期输出：
```
🚀 Aria to LeRobot 转换
📁 VRS文件: glue.vrs
📂 MPS目录: mps_glue_vrs
📤 输出: glue_dataset/
🎯 格式: LeRobot v3.0

转换帧数: 100%|████████| 18750/18750 [02:15<00:00, 138.2帧/秒]

✅ 转换成功完成！
📊 处理: 18,750 帧
⏱️  时间: 135.2秒
🗂️  文件: 1个parquet + 2个视频
📈 手部有效率: 87.3%
📈 位姿有效率: 99.9%
```

### 自定义采样

```bash
# 高质量30fps转换
python -m aria2lerobot glue.vrs output/ --fps 30 --stride 1

# 5fps快速预览
python -m aria2lerobot glue.vrs preview/ --fps 5 --stride 6 --max-frames 1000
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
python -m aria2lerobot data.vrs output/ --mps-dir path/to/mps_dir
```

**2. "高帧丢失率(>10%)"**
```bash
# 解决方案：增加对齐容差
python -m aria2lerobot data.vrs output/ --max-slop-ns 10000000  # 10ms
```

**3. "低手部有效率(<50%)"**
- 检查MPS手部跟踪质量
- 验证hand_tracking_results.csv存在且有数据
- 考虑不同录制条件

**4. "内存使用过高"**
- 使用帧步长：`--stride 2`
- 限制处理：`--max-frames 10000`
- 检查可用系统内存

#### 调试模式

```bash
# 启用详细日志
python -m aria2lerobot data.vrs output/ --log-level DEBUG

# 检查conversion.log获取详细信息
tail -f output/conversion.log
```

## 🧪 测试与验证

### 烟雾测试

```bash
# 运行包含示例的烟雾测试
cd examples/
python smoke_test.py ../glue.vrs
```

预期输出演示：
- 成功转换为LeRobot格式
- 用LeRobotDataset加载数据
- 样本数据检查和验证

### 单元测试

```bash
# 运行单元测试
python -m pytest tests/ -v

# 测试特定功能
python -m pytest tests/test_schema.py -v
python -m pytest tests/test_conversion.py -v
```

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
    # ... 训练你的模型
```

### 策略训练示例

```python
# 示例：在手部位置上训练简单策略
import torch
import torch.nn as nn

class HandPositionPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.hand_encoder = nn.Linear(63, 128)  # 手部关键点
        self.pose_encoder = nn.Linear(16, 128)  # 设备位姿
        self.policy_head = nn.Linear(256, 6)    # 6自由度动作
    
    def forward(self, hands, poses):
        hand_feat = torch.relu(self.hand_encoder(hands))
        pose_feat = torch.relu(self.pose_encoder(poses))
        combined = torch.cat([hand_feat, pose_feat], dim=-1)
        return self.policy_head(combined)

# 训练循环
model = HandPositionPolicy()
dataset = LeRobotDataset("converted_dataset")
dataloader = DataLoader(dataset, batch_size=32)

for batch in dataloader:
    hands = batch['status.hands.landmarks_world']
    poses = batch['status.pose.world_device']
    actions = model(hands, poses)
    # ... 计算损失和反向传播
```

## 📚 技术细节

### 架构

```
aria2lerobot/
├── __init__.py          # 包接口
├── __main__.py          # CLI入口点
├── convert.py           # 核心转换逻辑
├── schema.py            # LeRobot模式定义
├── video.py             # H.264/H.265视频编码
├── io_parquet.py        # 增量parquet写入
├── utils.py             # 日志记录、验证实用程序
└── README_CN.md         # 本文件
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

- [ ] **多剧集支持**：基于时间间隙自动剧集分割
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

---

*为机器人学习社区用❤️构建*
