#!/bin/bash
# 测试Aria触觉数据集成完整流程

set -e  # 遇到错误立即退出

echo "========================================="
echo "Aria触觉数据集成测试"
echo "========================================="

# 配置路径
VRS_FILE="/Volumes/BU100/Aria_data/glue.vrs"
HAPTIC_CSV="/Volumes/BU100/Aria_data/sensor_data/sequence_2_20250927_203243.csv"
OUTPUT_DIR="/Volumes/BU100/Aria_data/test_output"
TIME_OFFSET=1.31  # 从对齐算法得到的最优偏移

# 检查输入文件
echo ""
echo "1. 检查输入文件..."
if [ ! -f "$VRS_FILE" ]; then
    echo "❌ VRS文件不存在: $VRS_FILE"
    exit 1
fi

if [ ! -f "$HAPTIC_CSV" ]; then
    echo "❌ 触觉CSV文件不存在: $HAPTIC_CSV"
    exit 1
fi

echo "✅ 输入文件检查完成"
echo "   VRS: $VRS_FILE"
echo "   CSV: $HAPTIC_CSV"

# 创建输出目录
echo ""
echo "2. 创建输出目录..."
mkdir -p "$OUTPUT_DIR"
echo "✅ 输出目录: $OUTPUT_DIR"

# 步骤1: 生成Parquet数据集
echo ""
echo "3. 生成Parquet数据集（含触觉数据）..."
python3 convert_with_haptics.py \
    "$VRS_FILE" \
    "$OUTPUT_DIR/dataset" \
    --haptic-csv "$HAPTIC_CSV" \
    --haptic-offset $TIME_OFFSET \
    --fps 10 \
    --size 480 480

if [ $? -ne 0 ]; then
    echo "❌ Parquet数据集生成失败"
    exit 1
fi

echo "✅ Parquet数据集生成完成"

# 检查生成的文件
echo ""
echo "4. 检查生成的文件..."
if [ -f "$OUTPUT_DIR/dataset/data.parquet" ]; then
    echo "✅ data.parquet 已生成"
else
    echo "❌ data.parquet 未生成"
    exit 1
fi

if [ -f "$OUTPUT_DIR/dataset/metadata.json" ]; then
    echo "✅ metadata.json 已生成"
    cat "$OUTPUT_DIR/dataset/metadata.json"
else
    echo "❌ metadata.json 未生成"
fi

# 步骤2: 生成可视化视频
echo ""
echo "5. 生成可视化视频（含触觉指示条）..."
python3 visualize_with_haptics.py \
    "$VRS_FILE" \
    "$OUTPUT_DIR/visualization_with_haptics.mp4" \
    --haptic-csv "$HAPTIC_CSV" \
    --haptic-offset $TIME_OFFSET \
    --max-force 150 \
    --max-frames 300 \
    --stride 1

if [ $? -ne 0 ]; then
    echo "❌ 可视化视频生成失败"
    exit 1
fi

echo "✅ 可视化视频生成完成"

# 检查视频文件
if [ -f "$OUTPUT_DIR/visualization_with_haptics.mp4" ]; then
    echo "✅ 视频文件已生成: $OUTPUT_DIR/visualization_with_haptics.mp4"
    file_size=$(du -h "$OUTPUT_DIR/visualization_with_haptics.mp4" | cut -f1)
    echo "   文件大小: $file_size"
else
    echo "❌ 视频文件未生成"
    exit 1
fi

# 完成
echo ""
echo "========================================="
echo "✅ 测试完成！"
echo "========================================="
echo "生成的文件:"
echo "  1. Parquet数据集: $OUTPUT_DIR/dataset/"
echo "  2. 可视化视频: $OUTPUT_DIR/visualization_with_haptics.mp4"
echo ""
echo "使用方法:"
echo "  - 查看视频: open '$OUTPUT_DIR/visualization_with_haptics.mp4'"
echo "  - 加载数据: pd.read_parquet('$OUTPUT_DIR/dataset/data.parquet')"