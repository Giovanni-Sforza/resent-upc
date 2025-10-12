# 干涉条纹图像四分类项目

基于深度学习的干涉条纹图像自动分类系统，使用ResNet34预训练模型和判别性学习率策略。

## 项目特点

- 🔥 **GPU加速训练**：支持NVIDIA 3090等CUDA设备
- 📊 **实时监控**：TensorBoard可视化训练过程
- 🎯 **判别性学习率**：不同层使用不同学习率优化
- 📁 **结构化管理**：实验结果统一组织管理
- 🔧 **配置驱动**：通过YAML文件灵活配置实验参数
- 📈 **完整评估**：准确率、混淆矩阵、分类报告等

## 项目结构

```
interference_classification/
├── train.py              # 主训练脚本
├── inference.py          # 推理脚本
├── utils.py              # 工具函数
├── config.yaml           # 配置文件
├── requirements.txt      # 依赖包列表
├── README.md             # 项目说明
├── data/                 # 数据目录
│   └── interference_images/
│       ├── train/        # 训练集
│       └── val/          # 验证集
└── experiments/          # 实验输出目录
    └── resnet34_test1/   # 具体实验文件夹
        ├── logs/         # TensorBoard日志
        ├── checkpoints/  # 模型检查点
        └── training.log  # 训练日志
```

## 环境配置

### 1. Python环境

建议使用Python 3.7+版本：

```bash
# 创建虚拟环境
conda create -n interference_cls python=3.8
conda activate interference_cls

# 或者使用venv
python -m venv interference_cls
source interference_cls/bin/activate  # Linux/Mac
# interference_cls\Scripts\activate  # Windows
```

### 2. 安装依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 如果使用CUDA，请根据您的CUDA版本安装对应的PyTorch
# 例如CUDA 11.1：
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```

### 3. 数据准备

数据应该组织为以下结构：

```
data/interference_images/
├── train/
│   ├── class0_001.npy
│   ├── class0_002.npy
│   ├── class1_001.npy
│   ├── class1_002.npy
│   ├── class2_001.npy
│   ├── class2_002.npy
│   ├── class3_001.npy
│   └── class3_002.npy
└── val/
    ├── class0_101.npy
    ├── class1_101.npy
    ├── class2_101.npy
    └── class3_101.npy
```

**数据要求：**
- 格式：`.npy`文件
- 尺寸：`645x645`单通道数组
- 命名：文件名包含类别信息（class0, class1, class2, class3）

## 使用说明

### 1. 配置实验

编辑`config.yaml`文件设置实验参数：

```yaml
# 基本配置
experiment_name: "resnet34_test1"
output_dir: "experiments"
device: "auto"  # 'cuda', 'cpu', 或 'auto'

# 数据配置
data_path: "data/interference_images"
batch_size: 32
epochs: 100

# 判别性学习率配置
use_discriminative_lr: true
learning_rates:
  classifier: 0.001      # 分类头学习率
  layer_decay: 0.1       # 层间衰减因子
```

### 2. 开始训练

```bash
python train.py
```

训练过程中会：
- 自动创建实验输出目录
- 保存最佳模型检查点
- 记录TensorBoard日志
- 生成混淆矩阵和评估指标

### 3. 监控训练

启动TensorBoard查看训练进度：

```bash
tensorboard --logdir experiments/resnet34_test1/logs
```

然后在浏览器中访问：`http://localhost:6006`

### 4. 模型推理

#### 单张图像推理

```bash
python inference.py \
    --checkpoint experiments/resnet34_test1/checkpoints/best_model.pth \
    --single_image data/test/class0_001.npy \
    --config config.yaml
```

#### 批量推理

```bash
python inference.py \
    --checkpoint experiments/resnet34_test1/checkpoints/best_model.pth \
    --data_path data/test/ \
    --output inference_results.csv \
    --batch_size 32 \
    --has_labels \
    --generate_samples \
    --config config.yaml
```

## 核心技术特性

### 1. 判别性学习率策略

不同层使用不同的学习率以优化迁移学习效果：

- **分类头 (fc层)**: 1e-3 (最高学习率)
- **高层特征 (layer4)**: 1e-4
- **中层特征 (layer3)**: 1e-5  
- **底层特征 (conv1, bn1, layer1, layer2)**: 1e-6 (最低学习率)

### 2. 数据预处理流程

```
原始数据 (645×645, 单通道) 
    ↓
缩放到 (224×224, 单通道)
    ↓
复制为 (224×224, 3通道)
    ↓
ImageNet标准化
```

### 3. 模型架构

- **基础模型**: ResNet34 (ImageNet预训练)
- **分类头**: 全连接层 (512 → 4)
- **激活函数**: ReLU
- **输出**: 4类概率分布

## 实验管理

### 目录结构说明

每个实验会在`experiments/`下创建独立文件夹：

```
experiments/resnet34_test1/
├── logs/
│   └── events.out.tfevents.*    # TensorBoard日志
├── checkpoints/
│   ├── best_model.pth           # 最佳模型
│   ├── checkpoint_epoch_10.pth  # 定期检查点
│   └── confusion_matrix_*.png   # 混淆矩阵图像
├── training.log                 # 训练日志文件
└── final_config.yaml           # 最终配置备份
```

### 检查点内容

模型检查点包含：
- 模型权重 (`model_state_dict`)
- 优化器状态 (`optimizer_state_dict`)
- 训练轮次 (`epoch`)
- 最佳验证准确率 (`best_val_acc`)
- 实验配置 (`config`)

## 性能优化

### GPU加速配置

```yaml
device: "auto"              # 自动检测CUDA
pin_memory: true            # 加速数据传输
num_workers: 4              # 多进程数据加载
```

### 内存优化

- 使用`pin_memory`加速CPU到GPU数据传输
- 合理设置`batch_size`避免内存溢出
- 梯度累积支持大批次训练

## 故障排除

### 常见问题

1. **CUDA内存不足**
   ```
   解决方案：减小batch_size或使用梯度累积
   ```

2. **数据加载错误**
   ```bash
   # 检查数据完整性
   python -c "from utils import check_data_integrity; check_data_integrity('data/interference_images')"
   ```

3. **模型加载失败**
   ```
   确保使用正确的PyTorch版本和检查点文件
   ```

### 调试模式

在`config.yaml`中启用调试模式：

```yaml
debug:
  enabled: true
  max_batches_per_epoch: 10
  profile: false
```

## API参考

### 主要类和函数

#### `InterferenceDataset`
```python
dataset = InterferenceDataset(
    data_path="data/train",
    transform=PreprocessTransform(),
    split="train"
)
```

#### `PreprocessTransform`
```python
transform = PreprocessTransform()
# 将645x645单通道转为224x224三通道并标准化
```

#### `create_model`
```python
model = create_model(num_classes=4)
# 创建ResNet34四分类模型
```

#### `setup_discriminative_lr`
```python
param_groups = setup_discriminative_lr(
    model, 
    base_lr=1e-3, 
    layer_lr_decay=0.1
)
```

## 扩展和自定义

### 添加新的数据增强

```python
class CustomTransform:
    def __init__(self):
        # 只使用不破坏物理模式的变换
        self.brightness = transforms.ColorJitter(brightness=0.1)
        
    def __call__(self, image):
        # 自定义增强逻辑
        return self.brightness(image)
```

### 使用不同的预训练模型

```python
# 在create_model函数中修改
model = models.resnet50(pretrained=True)  # 使用ResNet50
model = models.efficientnet_b0(pretrained=True)  # 使用EfficientNet
```

### 自定义评估指标

```python
def custom_metric(y_true, y_pred):
    # 实现自定义指标
    return metric_value
```

## 实验记录

### 建议的实验流程

1. **基线实验**: 使用默认配置建立基线
2. **学习率调优**: 尝试不同的学习率组合
3. **批次大小优化**: 根据硬件调整批次大小
4. **模型对比**: 尝试不同的预训练模型
5. **集成学习**: 组合多个模型的预测结果

### 结果记录模板

```
实验名称: resnet34_test1
配置参数:
- 模型: ResNet34
- 批次大小: 32
- 学习率: 判别性 (1e-3, 0.1衰减)
- 训练轮次: 100

结果:
- 最佳验证准确率: XX.XX%
- 训练时间: XX小时
- GPU内存使用: XX GB
- 备注: 
```

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 贡献指南

欢迎提交Issue和Pull Request！

1. Fork项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

---

**注意**: 请确保您的数据符合项目要求，并根据实际硬件配置调整训练参数。
