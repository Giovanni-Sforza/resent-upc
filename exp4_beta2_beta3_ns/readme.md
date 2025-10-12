# 多任务深度学习模型测试程序使用指南

## 概述

这个测试程序专门用于评估已训练好的多任务深度学习模型，支持：
- 模型性能评估（回归 + 分类）
- ResNet特征提取和保存
- Grad-CAM可解释性分析
- 详细的结果可视化

## 文件结构

```
project/
├── train2.py    # 训练程序（需要导入）
├── test_multitask.py     # 测试程序
├── config2.yaml                   # 主配置文件
├── config_test.yaml       # 测试配置示例
└── experiments/
    └── [experiment_name]/
        ├── checkpoints/
        │   └── best_multitask_gradnorm_model.pth
        └── test_results/           # 测试结果输出目录
            ├── test.log            # 测试日志
            ├── test_predictions.npz # 预测结果
            ├── test_results.png    # 结果可视化图
            ├── features/           # 特征保存目录
            │   └── extracted_features.npz
            └── gradcam/           # Grad-CAM分析结果
                ├── classification/
                └── regression/
```

## 配置文件设置

### 1. 基础配置

```yaml
# 设置为测试模式
mode: "test"

# 指定测试数据路径
test_config:
  test_data_path: "/path/to/test/data"
  model_checkpoint: "/path/to/best_model.pth"
```

### 2. 特征保存配置

```yaml
test_config:
  save_features:
    enabled: true                    # 开启特征保存
    output_dir: "features"          # 输出目录名
    batch_size: 8                   # 测试批次大小
    include_processed_images: true  # 保存预处理后的图像
    include_original_paths: true    # 保存原始文件路径
```

启用特征保存后，程序会保存：
- `features`: ResNet骨干网络提取的特征向量
- `processed_images`: 经过预处理的输入图像
- `file_paths`: 原始数据文件路径
- 模型的预测结果和真实标签

### 3. Grad-CAM配置

```yaml
interpretability:
  gradcam:
    enabled: true                   # 开启Grad-CAM分析
    target_layers: ["layer4"]       # 目标层（ResNet层名）
    num_samples: 50                 # 分析的样本数量
    save_heatmaps: true            # 保存热力图
    save_overlay: true             # 保存叠加图
    colormap: "jet"                # 颜色方案
    
    task_specific:
      regression: true             # 分析回归任务
      classification: true         # 分析分类任务
```

## 运行步骤

### 1. 准备环境

确保安装了所有依赖包：
```bash
pip install torch torchvision opencv-python matplotlib seaborn scikit-learn pyyaml tqdm
```

### 2. 配置文件准备

复制并修改配置文件：
```bash
cp config2.yaml test_config.yaml
```

修改关键配置项：
- `mode: "test"`
- `test_config.model_checkpoint`: 指向训练好的模型
- `test_config.test_data_path`: 指向测试数据
- 根据需要调整特征保存和Grad-CAM设置

### 3. 运行测试

```bash
python test_multitask_gradnorm.py
```

程序会自动读取 `config2.yaml` 配置文件。如果要使用其他配置文件，需要修改代码中的文件名。

## 输出结果

### 1. 控制台输出

```
使用设备: cuda
成功加载模型权重，来自epoch 150
最佳验证损失: 0.123456
测试数据集大小: 1000
开始测试模型...
100%|██████████| 125/125 [01:30<00:00,  1.38it/s]
特征和数据已保存到: .../features/extracted_features.npz
开始Grad-CAM分析...
Grad-CAM分析进度: 10/50
...
测试完成!
```

### 2. 保存的文件

#### a) `test_predictions.npz`
包含完整的预测结果和评估指标：
```python
import numpy as np
data = np.load('test_predictions.npz', allow_pickle=True)

# 可用的键值
print(data.files)
# ['reg_predictions', 'reg_labels', 'cls_predictions', 'cls_labels', 
#  'file_paths', 'processed_images', 'features', 'regression_mse', ...]
```

#### b) `features/extracted_features.npz`
包含ResNet提取的特征：
```python
features_data = np.load('extracted_features.npz', allow_pickle=True)

# ResNet特征 (N, 2048) for ResNet50
features = features_data['features']  

# 预处理后的图像 (N, 3, 224, 224)
images = features_data['processed_images']  

# 原始文件路径
paths = features_data['file_paths']

# 预测结果
reg_preds = features_data['reg_predictions']
cls_preds = features_data['cls_predictions']
```

#### c) `test_results.png`
包含6个子图的综合结果可视化：
- Beta2预测 vs 真实值散点图
- Beta3预测 vs 真实值散点图  
- (Beta2, Beta3)空间中的预测分布
- 分类混淆矩阵
- 类别分布对比
- 性能指标总结

#### d) Grad-CAM结果
在 `gradcam/` 目录下：
- `classification/`: 分类任务的可解释性分析
- `regression/`: 回归任务的可解释性分析（分别针对beta2和beta3）

每个样本生成3张图：
- 原始图像
- Grad-CAM热力图  
- 热力图叠加图

## 高级用法

### 1. 批量处理多个模型

可以修改配置文件中的模型路径，对比不同模型的性能：

```python
models_to_test = [
    'model_epoch_100.pth',
    'model_epoch_150.pth', 
    'best_model.pth'
]

for model_path in models_to_test:
    # 更新配置并运行测试
    pass
```

### 2. 特定样本分析

如果只想分析特定的样本，可以修改 `num_samples` 参数或在数据集中添加过滤逻辑。

### 3. 自定义Grad-CAM目标层

根据ResNet架构选择不同的目标层：
```yaml
target_layers: ["layer3", "layer4"]  # 分析多个层
# 或者
target_layers: ["layer4.2.conv2"]    # 分析具体子层
```

### 4. 特征的后续使用

提取的特征可以用于：
- 降维可视化 (t-SNE, UMAP)
- 聚类分析
- 相似性搜索
- 迁移学习的初始化

```python
# 示例：使用提取的特征进行t-SNE可视化
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 加载特征
data = np.load('features/extracted_features.npz')
features = data['features']
labels = data['cls_labels']

# t-SNE降维
tsne = TSNE(n_components=2, random_state=42)
features_2d = tsne.fit_transform(features)

# 可视化
plt.figure(figsize=(10, 8))
for class_id in range(3):
    mask = labels == class_id
    plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                label=f'Class {class_id}', alpha=0.6)
plt.legend()
plt.title('t-SNE Visualization of ResNet Features')
plt.show()
```

## 故障排除

### 1. 常见错误

#### a) 模型加载失败
```
FileNotFoundError: 模型checkpoint不存在
```
**解决方案**：检查 `model_checkpoint` 路径是否正确

#### b) 配置不匹配
```
RuntimeError: size mismatch for feature_proj.0.weight
```
**解决方案**：确保测试配置中的模型参数与训练时完全一致

#### c) CUDA内存不足
```
RuntimeError: CUDA out of memory
```
**解决方案**：
- 减小 `batch_size`
- 减少 `num_samples` for Grad-CAM
- 设置 `device: "cpu"`

### 2. 性能优化

#### a) 加速测试
```yaml
test_config:
  save_features:
    batch_size: 32              # 增大批次大小
    include_processed_images: false  # 如果不需要图像可视化
    
interpretability:
  gradcam:
    num_samples: 20             # 减少分析样本数量
```

#### b) 减少内存占用
```yaml
num_workers: 1                  # 减少数据加载进程
interpretability:
  gradcam:
    enabled: false              # 关闭Grad-CAM分析
```

### 3. 自定义修改

#### a) 添加新的评估指标
在 `evaluate_results()` 函数中添加：
```python
# 示例：添加平均绝对百分比误差（MAPE）
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape_beta2 = mean_absolute_percentage_error(reg_labels[:, 0], reg_preds[:, 0])
metrics['mape_beta2'] = mape_beta2
```

#### b) 修改Grad-CAM可视化
在 `save_gradcam_visualization()` 函数中可以：
- 改变颜色映射
- 调整热力图阈值
- 添加更多统计信息

#### c) 扩展特征保存
在 `save_features_and_data()` 函数中可以添加：
- 中间层特征
- 注意力权重
- 梯度信息

## 注意事项

1. **数据一致性**：确保测试数据的预处理流程与训练时完全一致
2. **模型兼容性**：测试程序需要与训练程序中的模型定义保持同步
3. **内存管理**：处理大型数据集时注意内存使用，适当调整批次大小
4. **结果解释**：Grad-CAM结果需要结合领域知识进行解释，热力图高亮区域不一定代表重要特征
5. **文件权限**：确保输出目录具有写权限

## 扩展功能建议

1. **添加更多可解释性方法**：
   - Integrated Gradients
   - LIME (Local Interpretable Model-Agnostic Explanations)
   - SHAP (SHapley Additive exPlanations)

2. **批量评估**：
   - 支持多个数据集的批量测试
   - 不同模型的性能对比分析

3. **交互式可视化**：
   - 使用Plotly创建交互式图表
   - 集成Tensorboard进行结果展示

4. **自动化报告**：
   - 生成PDF格式的测试报告
   - 集成统计分析和假设检验

这个测试程序为你的多任务深度学习模型提供了全面的评估和分析功能，可以帮助你更好地理解模型的性能和行为特征。