#!/bin/bash

# 干涉条纹图像四分类实验运行脚本
# 使用方法: bash run_experiment.sh [experiment_name]

set -e  # 遇到错误时退出

# 默认实验名称
EXPERIMENT_NAME=${1:-"resnet34_$(date +%Y%m%d_%H%M%S)"}
CONFIG_FILE="config.yaml"
LOG_FILE="experiment_${EXPERIMENT_NAME}.log"

echo "========================================"
echo "干涉条纹图像四分类实验"
echo "实验名称: ${EXPERIMENT_NAME}"
echo "开始时间: $(date)"
echo "========================================"

# 检查Python环境
echo "检查Python环境..."
python --version
echo "检查PyTorch..."
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')"

if [ -f "${CONFIG_FILE}" ]; then
    echo "使用配置文件: ${CONFIG_FILE}"
else
    echo "错误: 配置文件 ${CONFIG_FILE} 不存在!"
    exit 1
fi

# 更新配置文件中的实验名称
echo "更新实验名称到配置文件..."
sed -i.bak "s/experiment_name: .*/experiment_name: \"${EXPERIMENT_NAME}\"/" ${CONFIG_FILE}

# 检查数据目录
DATA_PATH=$(python -c "import yaml; config=yaml.safe_load(open('${CONFIG_FILE}')); print(config['data_path'])")
if [ ! -d "${DATA_PATH}" ]; then
    echo "错误: 数据目录 ${DATA_PATH} 不存在!"
    exit 1
fi

echo "数据路径: ${DATA_PATH}"

# 检查数据完整性
echo "检查数据完整性..."
python -c "
from utils import check_data_integrity
if not check_data_integrity('${DATA_PATH}'):
    exit(1)
"

# 显示数据集统计信息
echo "数据集统计信息:"
python -c "from utils import analyze_dataset; analyze_dataset('${DATA_PATH}')"

# 创建实验输出目录
OUTPUT_DIR=$(python -c "import yaml; config=yaml.safe_load(open('${CONFIG_FILE}')); print(f\"{config['output_dir']}/{config['experiment_name']}\")")
mkdir -p "${OUTPUT_DIR}"
echo "实验输出目录: ${OUTPUT_DIR}"

# 备份配置文件
cp ${CONFIG_FILE} ${OUTPUT_DIR}/config_backup.yaml
echo "配置文件已备份到: ${OUTPUT_DIR}/config_backup.yaml"

# 开始训练
echo "开始训练..."
echo "训练日志将保存到: ${LOG_FILE}"

# 运行训练并记录日志
python train.py 2>&1 | tee ${LOG_FILE}

TRAINING_EXIT_CODE=$?

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "训练成功完成!"
    
    # 移动日志文件到实验目录
    mv ${LOG_FILE} ${OUTPUT_DIR}/
    
    # 生成训练历史图表
    echo "生成训练历史图表..."
    python -c "
from utils import plot_training_history
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
try:
    plot_training_history('${OUTPUT_DIR}/logs', '${OUTPUT_DIR}/training_history.png')
    print('训练历史图表已保存')
except Exception as e:
    print(f'生成训练历史图表失败: {e}')
"
    
    # 模型性能统计
    echo "计算模型参数统计..."
    python -c "
from train import create_model
from utils import calculate_model_size
model = create_model(4)
calculate_model_size(model)
" >> ${OUTPUT_DIR}/model_info.txt
    
    # 寻找最佳检查点
    BEST_CHECKPOINT="${OUTPUT_DIR}/checkpoints/best_model.pth"
    if [ -f "${BEST_CHECKPOINT}" ]; then
        echo "最佳模型检查点: ${BEST_CHECKPOINT}"
        
        # 在验证集上进行最终评估
        if [ -d "${DATA_PATH}/val" ]; then
            echo "在验证集上进行最终评估..."
            python inference.py \
                --checkpoint "${BEST_CHECKPOINT}" \
                --data_path "${DATA_PATH}/val" \
                --output "${OUTPUT_DIR}/final_validation_results.csv" \
                --has_labels \
                --generate_samples \
                --config ${CONFIG_FILE} >> ${OUTPUT_DIR}/final_evaluation.log 2>&1
            
            echo "最终评估结果已保存到: ${OUTPUT_DIR}/final_evaluation.log"
        fi
    else
        echo "警告: 未找到最佳模型检查点!"
    fi
    
    # 生成实验报告
    echo "生成实验报告..."
    cat > ${OUTPUT_DIR}/experiment_report.md << EOF
# 实验报告: ${EXPERIMENT_NAME}

## 基本信息
- 实验名称: ${EXPERIMENT_NAME}
- 开始时间: $(date)
- 配置文件: config_backup.yaml
- 训练状态: 成功完成

## 文件结构
\`\`\`
${OUTPUT_DIR}/
├── logs/                    # TensorBoard日志
├── checkpoints/            # 模型检查点
│   └── best_model.pth     # 最佳模型
├── config_backup.yaml     # 配置文件备份
├── training_history.png   # 训练历史图表
├── model_info.txt         # 模型参数统计
├── final_evaluation.log   # 最终评估日志
└── experiment_report.md   # 本报告
\`\`\`

## 使用方法

### 推理单张图像
\`\`\`bash
python inference.py \\
    --checkpoint ${BEST_CHECKPOINT} \\
    --single_image path/to/image.npy \\
    --config config.yaml
\`\`\`

### 批量推理
\`\`\`bash
python inference.py \\
    --checkpoint ${BEST_CHECKPOINT} \\
    --data_path path/to/test/data \\
    --output results.csv \\
    --has_labels \\
    --generate_samples
\`\`\`

## TensorBoard查看
\`\`\`bash
tensorboard --logdir ${OUTPUT_DIR}/logs
\`\`\`

EOF
    
    echo "实验报告已生成: ${OUTPUT_DIR}/experiment_report.md"
    
else
    echo "训练失败! 退出码: ${TRAINING_EXIT_CODE}"
    mv ${LOG_FILE} ${OUTPUT_DIR}/ 2>/dev/null || true
fi

# 恢复原始配置文件
if [ -f "${CONFIG_FILE}.bak" ]; then
    mv ${CONFIG_FILE}.bak ${CONFIG_FILE}
fi

echo "========================================"
echo "实验完成时间: $(date)"
echo "实验目录: ${OUTPUT_DIR}"
echo "========================================"

# 显示磁盘使用情况
echo "实验文件大小:"
du -sh ${OUTPUT_DIR}

exit $TRAINING_EXIT_CODE