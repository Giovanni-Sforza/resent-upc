import os
import yaml
import random
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error, classification_report

# 导入 gpytorch (如果选择GP模式)
try:
    import gpytorch
    GP_AVAILABLE = True
except ImportError:
    GP_AVAILABLE = False

# ===================================================================
# 新增：图像对称分割工具类
# ===================================================================

class SymmetricSplitter:
    """对称分割工具：将图像按中轴线分割并生成左右对称图像"""
    def __init__(self, enabled=True):
        self.enabled = enabled
    
    def split_and_mirror(self, image_tensor):
        """
        将图像按中轴线分割成左右两部分，并各自镜像对称
        
        Args:
            image_tensor: 形状为 [C, H, W] 或 [H, W] 的张量
        
        Returns:
            left_image: 左半部分镜像后的完整图像
            right_image: 右半部分镜像后的完整图像
        """
        if not self.enabled:
            return image_tensor, image_tensor
        
        # 处理维度
        if image_tensor.dim() == 2:
            image_tensor = image_tensor.unsqueeze(0)  # [H, W] -> [1, H, W]
        
        C, H, W = image_tensor.shape
        mid_h = H // 2
        
        # 分割左右两部分
        left_half = image_tensor[:, :mid_h, : ]  # [C, H, W/2]
        right_half = image_tensor[:, mid_h:, :]  # [C, H, W/2]
        
        # 镜像对称
        left_mirrored = torch.flip(left_half, dims=[1])  # 水平翻转
        right_mirrored = torch.flip(right_half, dims=[1])
        
        # 拼接成完整图像
        left_image = torch.cat([left_half, left_mirrored], dim=1)  # [C, H, W]
        right_image = torch.cat([right_mirrored, right_half], dim=1)
        
        return left_image, right_image
    
    def __call__(self, image_tensor):
        return self.split_and_mirror(image_tensor)


# ===================================================================
# 定义一个自定义的Logger类，同时输出到文件和控制台
# ===================================================================

class CustomLogger:
    """自定义日志记录器，同时输出到文件和控制台"""
    def __init__(self, log_file):
        self.log_file = log_file
        self.terminal = sys.stdout
        
        # 确保log文件所在目录存在
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 打开日志文件
        self.log = open(log_file, 'w', encoding='utf-8')

    def write(self, message):
        # 写入到终端
        self.terminal.write(message)
        # 写入到文件
        self.log.write(message)
        # 立即刷新文件缓冲区
        self.log.flush()

    def flush(self):
        # 刷新终端和文件
        self.terminal.flush()
        self.log.flush()

    def close(self):
        # 关闭日志文件
        if hasattr(self, 'log'):
            self.log.close()


class GradNormWeights(nn.Module):
    """用于管理GradNorm可学习权重的模块"""
    def __init__(self, num_tasks):
        super().__init__()
        # 将权重初始化为1.0
        self.weights = nn.Parameter(torch.ones(num_tasks))
        self.num_tasks = num_tasks

    def forward(self):
        return self.weights
    
    def renormalize(self):
        """重新规范化权重"""
        renorm_weights = self.num_tasks * F.softmax(self.weights, dim=0)
        self.weights.data = renorm_weights.data


class GradNormTrainer:
    """GradNorm训练器，负责管理多任务权重的动态调整"""
    def __init__(self, model, num_tasks, alpha=1.5):
        self.model = model
        self.num_tasks = num_tasks
        self.alpha = alpha
        
        self.task_weights = GradNormWeights(num_tasks)
        self.shared_params = self._get_shared_parameters()
        
        self.initial_losses = None
        self.running_losses = None
        
    def _get_shared_parameters(self):
        """获取用于计算梯度的共享参数"""
        for name, module in self.model.feature_extractor.feature_proj.named_modules():
            if isinstance(module, nn.Linear):
                last_linear = module
        
        return last_linear.weight
    
    def compute_grad_norm(self, losses):
        """计算并应用GradNorm算法"""
        if self.initial_losses is None:
            self.initial_losses = losses.clone().detach()
            self.running_losses = losses.clone().detach()
        
        decay = 0.1
        self.running_losses = (1 - decay) * self.running_losses + decay * losses.detach()
        
        weights = self.task_weights()
        weighted_loss = torch.sum(weights * losses)
        
        self.shared_params.grad = None
        
        grad_norms = []
        for i, loss in enumerate(losses):
            task_grad = torch.autograd.grad(
                weights[i] * loss, 
                self.shared_params, 
                retain_graph=True,
                create_graph=True
            )[0]
            
            grad_norm = torch.norm(task_grad)
            grad_norms.append(grad_norm)
        
        grad_norms = torch.stack(grad_norms)
        
        loss_ratios = self.running_losses / self.initial_losses
        mean_grad_norm = torch.mean(grad_norms)
        
        relative_rates = torch.pow(loss_ratios, self.alpha)
        mean_relative_rate = torch.mean(relative_rates)
        target_grad_norms = mean_grad_norm * (relative_rates / mean_relative_rate)
        
        gradnorm_loss = torch.sum(torch.abs(grad_norms - target_grad_norms))
        
        return weighted_loss, gradnorm_loss
    
    def get_current_weights(self):
        """获取当前的任务权重"""
        return self.task_weights().detach().cpu().numpy()


# ===================================================================
# 1. 数据预处理模块
# ===================================================================

class CenterCropTensor:
    """对 PyTorch 张量进行中心裁剪"""
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, tensor):
        h, w = tensor.shape[-2:]
        th, tw = self.size

        if h < th or w < tw:
            raise ValueError(f"输入张量尺寸 ({h}x{w}) 小于目标裁剪尺寸 ({th}x{tw})。")

        start_h = (h - th) // 2
        start_w = (w - tw) // 2
        
        return tensor[..., start_h:start_h + th, start_w:start_w + tw]

    def __repr__(self):
        return self.__class__.__name__ + f'(size={self.size})'


class LogNormalization:
    """对数归一化变换"""
    def __init__(self, epsilon=1e-12, enabled=True):
        self.epsilon = epsilon
        self.enabled = enabled

    def __call__(self, tensor):
        if not self.enabled:
            return tensor
            
        tensor = torch.log1p(tensor + self.epsilon)
        
        min_val, max_val = torch.min(tensor), torch.max(tensor)
        if max_val > min_val:
            tensor = (tensor - min_val) / (max_val - min_val)
        
        return tensor

    def __repr__(self):
        return f"{self.__class__.__name__}(epsilon={self.epsilon}, enabled={self.enabled})"


class DataPreprocessor:
    """统一的数据预处理管道"""
    def __init__(self, config):
        self.config = config['preprocessing']
        self.transforms = self._build_transforms()

    def _build_transforms(self):
        transforms_list = []
        
        # 1. 中心裁剪 (如果启用)
        if self.config.get('center_crop', {}).get('enabled', False):
            crop_size = self.config['center_crop']['size']
            transforms_list.append(CenterCropTensor(crop_size))
            print(f"启用中心裁剪: {crop_size}")

        # 2. 对数归一化 (如果启用)
        if self.config.get('log_normalization', {}).get('enabled', False):
            epsilon = self.config['log_normalization'].get('epsilon', 1e-12)
            transforms_list.append(LogNormalization(epsilon=epsilon, enabled=True))
            print(f"启用对数归一化: epsilon={epsilon}")

        # 3. 调整大小
        if self.config.get('resize', {}).get('enabled', True):
            resize_dim = self.config['resize'].get('size', 224)
            transforms_list.append(transforms.Resize((resize_dim, resize_dim)))
            print(f"启用图像调整大小: {resize_dim}x{resize_dim}")

        # 4. 通道重复 (1通道 -> 3通道)
        transforms_list.append(self._repeat_channels)

        # 5. 标准化 (如果启用)
        if self.config.get('normalize', {}).get('enabled', True):
            mean = self.config['normalize'].get('mean', [0.485, 0.456, 0.406])
            std = self.config['normalize'].get('std', [0.229, 0.224, 0.225])
            transforms_list.append(transforms.Normalize(mean=mean, std=std))
            print(f"启用标准化: mean={mean}, std={std}")

        return transforms.Compose(transforms_list)

    def _repeat_channels(self, tensor):
        """将单通道图像重复为三通道"""
        if tensor.shape[0] == 1:
            return tensor.repeat(3, 1, 1)
        return tensor

    def __call__(self, image):
        return self.transforms(image)


# ===================================================================
# 2. 数据集类（修改为多任务 + 对称分割）
# ===================================================================

class MultiTaskInterferenceDataset(Dataset):
    """
    多任务干涉数据集类（支持对称分割）
    - 回归任务：预测beta2, beta3
    - 分类任务1：预测类别（class 0/1/2）
    - 分类任务2：预测左右（left/right）
    """
    def __init__(self, data_path, config, split='train'):
        self.data_path = Path(data_path)
        self.config = config
        self.split = split
        self.mode = config['inference_mode']
        
        # 初始化预处理器
        self.preprocessor = DataPreprocessor(config)
        
        # 初始化对称分割器
        self.use_symmetric_split = config.get('symmetric_split', {}).get('enabled', False)
        self.symmetric_splitter = SymmetricSplitter(enabled=self.use_symmetric_split)
        
        # 加载数据文件
        self.file_paths = sorted(list(self.data_path.glob('**/*.npy')))
        if not self.file_paths:
            raise ValueError(f"在路径 {self.data_path} 中没有找到任何.npy文件!")
            
        # 解析标签
        self.regression_labels, self.classification_labels = self._parse_labels()
        
        # 如果启用对称分割，每个样本会生成2个（左右）
        self.samples_per_image = 2 if self.use_symmetric_split else 1
        
        print(f"'{self.split}' 数据集初始化完成:")
        print(f"  - 原始文件数量: {len(self.file_paths)}")
        print(f"  - 对称分割: {'启用' if self.use_symmetric_split else '禁用'}")
        print(f"  - 实际样本数量: {len(self)}")
        print(f"  - 推理模式: {self.mode}")
        print(f"  - 类别数量: 3")
        print(f"  - 左右标签: {'启用' if self.use_symmetric_split else '禁用'}")

    def _parse_labels(self):
        """解析beta值标签和分类标签"""
        beta_pattern = re.compile(r"beta2_([\d.]+)_beta3_([\d.]+)")
        class_pattern = re.compile(r"class(\d)")
        
        beta_pairs = []
        class_labels = []
        
        for file_path in self.file_paths:
            # 解析beta值
            beta_match = beta_pattern.search(file_path.stem)
            if beta_match:
                beta2 = float(beta_match.group(1))
                beta3 = float(beta_match.group(2))
                beta_pairs.append((beta2, beta3))
            else:
                print(f"警告: 文件 {file_path.name} 无法匹配beta值, 已跳过。")
                continue
            
            # 解析类别标签
            class_match = class_pattern.search(file_path.stem)
            if class_match:
                class_id = int(class_match.group(1))
                if class_id not in [0, 1, 2]:
                    print(f"警告: 文件 {file_path.name} 包含无效的类别ID {class_id}, 已跳过。")
                    beta_pairs.pop()
                    continue
                class_labels.append(class_id)
            else:
                print(f"警告: 文件 {file_path.name} 无法匹配类别标签, 已跳过。")
                beta_pairs.pop()
                continue

        if not beta_pairs or not class_labels:
            raise ValueError(f"在路径 {self.data_path} 中没有找到任何有效的数据文件!")

        regression_labels = [torch.tensor([p[0], p[1]], dtype=torch.float32) for p in beta_pairs]
        
        print(f"多任务模式: 加载 {len(regression_labels)} 个样本")
        print(f"  - 回归标签: (beta2, beta3) 坐标")
        print(f"  - 分类标签分布: {dict(zip(*np.unique(class_labels, return_counts=True)))}")
        
        return regression_labels, class_labels

    def __len__(self):
        return len(self.regression_labels) * self.samples_per_image

    def __getitem__(self, idx):
        # 计算原始图像索引和左右标签
        if self.use_symmetric_split:
            image_idx = idx // 2
            side_label = idx % 2  # 0: left, 1: right
        else:
            image_idx = idx
            side_label = 0  # 不使用对称分割时，默认为0
        
        # 加载原始数据
        file_path = self.file_paths[image_idx]
        image_data = np.load(file_path)
        
        # 转换为PyTorch张量并增加通道维度
        image_tensor = torch.from_numpy(image_data.astype(np.float32)).unsqueeze(0)
        
        # 如果启用对称分割，进行分割处理
        if self.use_symmetric_split:
            left_image, right_image = self.symmetric_splitter(image_tensor)
            image_tensor = left_image if side_label == 0 else right_image
        
        # 应用预处理流程
        image_tensor = self.preprocessor(image_tensor)
        
        # 获取标签（回归和分类标签继承原始图像）
        regression_label = self.regression_labels[image_idx]
        classification_label = self.classification_labels[image_idx]
        
        return image_tensor, regression_label, classification_label, side_label


# ===================================================================
# 3. 多任务模型定义（新增左右分类头）
# ===================================================================

class ResNetFeatureExtractor(nn.Module):
    """ResNet特征提取器"""
    def __init__(self, feature_dim, dropout_rate=0.1):
        super().__init__()
        resnet = models.resnet34(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        self.feature_proj = nn.Sequential(
            nn.Linear(resnet.fc.in_features, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.feature_proj(x)
        return x


class MultiTaskResNetMLP(nn.Module):
    """ResNet + 多任务MLP模型（回归 + 类别分类 + 左右分类）"""
    def __init__(self, config):
        super().__init__()
        mlp_config = config['multitask_mlp_config']
        model_config = config.get('model_config', {})
        
        # 特征提取器
        feature_dim = mlp_config['feature_dim']
        dropout_rate = model_config.get('dropout_rate', 0.1)
        self.feature_extractor = ResNetFeatureExtractor(feature_dim, dropout_rate)
        
        # 共享特征层
        self.shared_layers = self._build_shared_layers(mlp_config, feature_dim)
        
        # 任务特定层
        shared_output_dim = mlp_config['shared_dims'][-1] if mlp_config['shared_dims'] else feature_dim
        self.regression_head = self._build_regression_head(mlp_config, shared_output_dim)
        self.classification_head = self._build_classification_head(mlp_config, shared_output_dim)
        
        # 新增：左右分类头
        self.use_side_classification = config.get('symmetric_split', {}).get('enabled', False)
        if self.use_side_classification:
            self.side_classification_head = self._build_side_classification_head(mlp_config, shared_output_dim)
        
        # 初始化权重
        self._initialize_weights()
    
    def _build_shared_layers(self, mlp_config, input_dim):
        """构建共享特征层"""
        if not mlp_config.get('shared_dims'):
            return nn.Identity()
            
        layers = []
        current_dim = input_dim
        
        for hidden_dim in mlp_config['shared_dims']:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(mlp_config.get('dropout', 0.1))
            ])
            current_dim = hidden_dim
        
        return nn.Sequential(*layers)
    
    def _build_regression_head(self, mlp_config, input_dim):
        """构建回归头"""
        layers = []
        current_dim = input_dim
        
        for hidden_dim in mlp_config['regression_dims']:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(mlp_config.get('dropout', 0.3))
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, 2))
        
        return nn.Sequential(*layers)
    
    def _build_classification_head(self, mlp_config, input_dim):
        """构建类别分类头"""
        layers = []
        current_dim = input_dim
        
        for hidden_dim in mlp_config['classification_dims']:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(mlp_config.get('dropout', 0.3))
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, 3))
        
        return nn.Sequential(*layers)
    
    def _build_side_classification_head(self, mlp_config, input_dim):
        """构建左右分类头"""
        layers = []
        current_dim = input_dim
        
        # 使用较简单的网络结构
        side_dims = mlp_config.get('side_classification_dims', [128, 64])
        
        for hidden_dim in side_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(mlp_config.get('dropout', 0.3))
            ])
            current_dim = hidden_dim
        
        # 输出层（2个类别：left/right）
        layers.append(nn.Linear(current_dim, 2))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 特征提取
        features = self.feature_extractor(x)
        
        # 共享特征
        shared_features = self.shared_layers(features)
        
        # 任务特定预测
        regression_output = self.regression_head(shared_features)
        classification_output = self.classification_head(shared_features)
        
        # 确保回归输出为正值
        regression_output = F.softplus(regression_output) + 1e-6
        
        # 左右分类
        if self.use_side_classification:
            side_output = self.side_classification_head(shared_features)
            return regression_output, classification_output, side_output
        else:
            return regression_output, classification_output, None


def create_model(config, device='cpu'):
    """模型创建工厂函数"""
    mode = config['inference_mode']
    
    if mode == 'multitask_mlp':
        return MultiTaskResNetMLP(config)
    else:
        raise ValueError(f"当前只支持 'multitask_mlp' 模式，收到: {mode}")


# ===================================================================
# 4. 多任务训练与验证函数（扩展到3个任务）
# ===================================================================

def setup_discriminative_lr_multitask(model, config):
    """为 MultiTaskResNetMLP 模型设置分层学习率"""
    lr_conf = config['learning_rates']
    base_lr = lr_conf['base']
    decay = lr_conf.get('layer_decay', 0.9)

    if not isinstance(model, MultiTaskResNetMLP):
        raise TypeError(f"此函数专为 MultiTaskResNetMLP 设计，但收到的模型类型为 {type(model)}")

    resnet_backbone = model.feature_extractor.features

    layer_groups = [
        list(resnet_backbone[0].parameters()) + list(resnet_backbone[1].parameters()) +
        list(resnet_backbone[4].parameters()) + list(resnet_backbone[5].parameters()),
        list(resnet_backbone[6].parameters()),
        list(resnet_backbone[7].parameters()),
    ]

    head_layers = list(model.feature_extractor.feature_proj.parameters()) + \
                  list(model.shared_layers.parameters()) + \
                  list(model.regression_head.parameters()) + \
                  list(model.classification_head.parameters())
    
    # 添加左右分类头的参数
    if model.use_side_classification:
        head_layers += list(model.side_classification_head.parameters())

    optimizer_params = [
        {'params': layer_groups[0], 'lr': base_lr * (decay ** 3)},
        {'params': layer_groups[1], 'lr': base_lr * (decay ** 2)},
        {'params': layer_groups[2], 'lr': base_lr * decay},
        {'params': head_layers, 'lr': base_lr}
    ]

    print("差分学习率设置成功:")
    print(f"  - 头部 (新层) 学习率: {base_lr:.6f}")
    print(f"  - ResNet 后期层学习率: {base_lr * decay:.6f}")
    print(f"  - ResNet 中期层学习率: {base_lr * (decay ** 2):.6f}")
    print(f"  - ResNet 早期层学习率: {base_lr * (decay ** 3):.6f}")
    
    return optimizer_params


def train_epoch_multitask_gradnorm(model, train_loader, regression_criterion, classification_criterion, 
                                  side_criterion, optimizer, gradnorm_optimizer, gradnorm_trainer, 
                                  device, epoch, config):
    """使用GradNorm的多任务训练一个epoch（支持3个任务）"""
    model.train()
    gradnorm_trainer.task_weights.train()
    
    use_side_classification = config.get('symmetric_split', {}).get('enabled', False)
    
    running_total_loss = 0.0
    running_reg_loss = 0.0
    running_cls_loss = 0.0
    running_side_loss = 0.0
    running_gradnorm_loss = 0.0
    correct_predictions = 0
    side_correct_predictions = 0
    total_samples = 0
    
    weight_history = []
    
    for i, batch in enumerate(train_loader):
        if use_side_classification:
            inputs, reg_labels, cls_labels, side_labels = batch
            side_labels = side_labels.to(device)
        else:
            inputs, reg_labels, cls_labels, _ = batch
            side_labels = None
        
        inputs = inputs.to(device)
        reg_labels = reg_labels.to(device)
        cls_labels = cls_labels.to(device)
        
        optimizer.zero_grad()
        gradnorm_optimizer.zero_grad()
        
        # 前向传播
        reg_outputs, cls_outputs, side_outputs = model(inputs)
        
        # 计算各任务损失
        reg_loss = regression_criterion(reg_outputs, reg_labels)
        cls_loss = classification_criterion(cls_outputs, cls_labels)
        
        if use_side_classification and side_outputs is not None:
            side_loss = side_criterion(side_outputs, side_labels)
            task_losses = torch.stack([reg_loss, cls_loss, side_loss])
        else:
            task_losses = torch.stack([reg_loss, cls_loss])
        
        # 使用GradNorm计算加权损失
        weighted_loss, gradnorm_loss = gradnorm_trainer.compute_grad_norm(task_losses)
        
        gradnorm_weight = config.get('gradnorm', {}).get('weight', 0.1)
        total_loss = weighted_loss + gradnorm_weight * gradnorm_loss
        
        # 反向传播和优化
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(gradnorm_trainer.task_weights.parameters(), max_norm=1.0)
        
        optimizer.step()
        gradnorm_optimizer.step()
        
        gradnorm_trainer.task_weights.renormalize()
        
        # 统计
        running_total_loss += total_loss.item()
        running_reg_loss += reg_loss.item()
        running_cls_loss += cls_loss.item()
        if use_side_classification and side_outputs is not None:
            running_side_loss += side_loss.item()
        running_gradnorm_loss += gradnorm_loss.item()
        
        # 分类准确率
        _, predicted = cls_outputs.max(1)
        correct = predicted.eq(cls_labels).sum().item()
        correct_predictions += correct
        
        # 左右分类准确率
        if use_side_classification and side_outputs is not None:
            _, side_predicted = side_outputs.max(1)
            side_correct = side_predicted.eq(side_labels).sum().item()
            side_correct_predictions += side_correct
        
        total_samples += cls_labels.size(0)
        
        # 记录权重
        current_weights = gradnorm_trainer.get_current_weights()
        weight_history.append(current_weights)
    
    # 计算平均值
    avg_total_loss = running_total_loss / len(train_loader)
    avg_reg_loss = running_reg_loss / len(train_loader)
    avg_cls_loss = running_cls_loss / len(train_loader)
    avg_side_loss = running_side_loss / len(train_loader) if use_side_classification else 0.0
    avg_gradnorm_loss = running_gradnorm_loss / len(train_loader)
    train_acc = 100. * correct_predictions / total_samples if total_samples > 0 else 0.0
    side_train_acc = 100. * side_correct_predictions / total_samples if (use_side_classification and total_samples > 0) else 0.0
    
    # 获取最终权重
    final_weights = gradnorm_trainer.get_current_weights()
    
    print(f"Epoch {epoch} - 训练完成:")
    print(f"  总损失: {avg_total_loss:.4f} | 回归损失: {avg_reg_loss:.4f} | 分类损失: {avg_cls_loss:.4f}")
    if use_side_classification:
        print(f"  左右分类损失: {avg_side_loss:.4f} | 左右分类准确率: {side_train_acc:.2f}%")
    print(f"  GradNorm损失: {avg_gradnorm_loss:.4f} | 分类准确率: {train_acc:.2f}%")
    if use_side_classification:
        print(f"  任务权重 - 回归: {final_weights[0]:.4f}, 分类: {final_weights[1]:.4f}, 左右: {final_weights[2]:.4f}")
    else:
        print(f"  任务权重 - 回归: {final_weights[0]:.4f}, 分类: {final_weights[1]:.4f}")
    
    metrics = {
        'total_loss': avg_total_loss,
        'regression_loss': avg_reg_loss,
        'classification_loss': avg_cls_loss,
        'side_loss': avg_side_loss,
        'gradnorm_loss': avg_gradnorm_loss,
        'train_accuracy': train_acc,
        'side_train_accuracy': side_train_acc,
        'task_weights': final_weights,
        'weight_history': np.array(weight_history)
    }
    
    return metrics


def validate_epoch_multitask_gradnorm(model, val_loader, regression_criterion, classification_criterion, 
                                     side_criterion, gradnorm_trainer, device, epoch, config):
    """使用GradNorm权重的多任务验证一个epoch（支持3个任务）"""
    model.eval()
    gradnorm_trainer.task_weights.eval()
    
    use_side_classification = config.get('symmetric_split', {}).get('enabled', False)
    
    running_total_loss = 0.0
    running_reg_loss = 0.0
    running_cls_loss = 0.0
    running_side_loss = 0.0
    
    all_reg_preds, all_reg_labels = [], []
    all_cls_preds, all_cls_labels = [], []
    all_side_preds, all_side_labels = [], []
    
    current_weights = gradnorm_trainer.get_current_weights()
    
    with torch.no_grad():
        for batch in val_loader:
            if use_side_classification:
                inputs, reg_labels, cls_labels, side_labels = batch
                side_labels = side_labels.to(device)
            else:
                inputs, reg_labels, cls_labels, _ = batch
                side_labels = None
            
            inputs = inputs.to(device)
            reg_labels = reg_labels.to(device)
            cls_labels = cls_labels.to(device)
            
            # 前向传播
            reg_outputs, cls_outputs, side_outputs = model(inputs)
            
            # 计算损失
            reg_loss = regression_criterion(reg_outputs, reg_labels)
            cls_loss = classification_criterion(cls_outputs, cls_labels)
            
            if use_side_classification and side_outputs is not None:
                side_loss = side_criterion(side_outputs, side_labels)
                total_loss = current_weights[0] * reg_loss + current_weights[1] * cls_loss + current_weights[2] * side_loss
                running_side_loss += side_loss.item()
            else:
                total_loss = current_weights[0] * reg_loss + current_weights[1] * cls_loss
            
            running_total_loss += total_loss.item()
            running_reg_loss += reg_loss.item()
            running_cls_loss += cls_loss.item()
            
            # 收集预测结果
            all_reg_preds.extend(reg_outputs.cpu().numpy())
            all_reg_labels.extend(reg_labels.cpu().numpy())
            
            _, predicted = cls_outputs.max(1)
            all_cls_preds.extend(predicted.cpu().numpy())
            all_cls_labels.extend(cls_labels.cpu().numpy())
            
            # 收集左右分类结果
            if use_side_classification and side_outputs is not None:
                _, side_predicted = side_outputs.max(1)
                all_side_preds.extend(side_predicted.cpu().numpy())
                all_side_labels.extend(side_labels.cpu().numpy())
    
    # 计算评估指标
    avg_total_loss = running_total_loss / len(val_loader)
    avg_reg_loss = running_reg_loss / len(val_loader)
    avg_cls_loss = running_cls_loss / len(val_loader)
    avg_side_loss = running_side_loss / len(val_loader) if use_side_classification else 0.0
    
    # 回归评估指标
    all_reg_preds = np.array(all_reg_preds)
    all_reg_labels = np.array(all_reg_labels)
    
    reg_mse = mean_squared_error(all_reg_labels, all_reg_preds)
    reg_mae = mean_absolute_error(all_reg_labels, all_reg_preds)
    
    mse_beta2 = mean_squared_error(all_reg_labels[:, 0], all_reg_preds[:, 0])
    mse_beta3 = mean_squared_error(all_reg_labels[:, 1], all_reg_preds[:, 1])
    mae_beta2 = mean_absolute_error(all_reg_labels[:, 0], all_reg_preds[:, 0])
    mae_beta3 = mean_absolute_error(all_reg_labels[:, 1], all_reg_preds[:, 1])
    
    r2_beta2 = r2_score(all_reg_labels[:, 0], all_reg_preds[:, 0])
    r2_beta3 = r2_score(all_reg_labels[:, 1], all_reg_preds[:, 1])
    
    # 分类评估指标
    val_acc = 100. * np.sum(np.array(all_cls_preds) == np.array(all_cls_labels)) / len(all_cls_labels)
    cm = confusion_matrix(all_cls_labels, all_cls_preds)
    
    # 左右分类评估指标
    side_val_acc = 0.0
    side_cm = None
    if use_side_classification and len(all_side_preds) > 0:
        side_val_acc = 100. * np.sum(np.array(all_side_preds) == np.array(all_side_labels)) / len(all_side_labels)
        side_cm = confusion_matrix(all_side_labels, all_side_preds)
    
    print(f"验证完成:")
    print(f"  总损失: {avg_total_loss:.4f} | 回归损失: {avg_reg_loss:.4f} | 分类损失: {avg_cls_loss:.4f}")
    if use_side_classification:
        print(f"  左右分类损失: {avg_side_loss:.4f} | 左右分类准确率: {side_val_acc:.2f}%")
    print(f"  回归 - MSE: {reg_mse:.6f}, MAE: {reg_mae:.6f}")
    print(f"  Beta2 - MSE: {mse_beta2:.6f}, MAE: {mae_beta2:.6f}, R²: {r2_beta2:.6f}")
    print(f"  Beta3 - MSE: {mse_beta3:.6f}, MAE: {mae_beta3:.6f}, R²: {r2_beta3:.6f}")
    print(f"  分类准确率: {val_acc:.2f}%")
    if use_side_classification:
        print(f"  当前任务权重 - 回归: {current_weights[0]:.4f}, 分类: {current_weights[1]:.4f}, 左右: {current_weights[2]:.4f}")
    else:
        print(f"  当前任务权重 - 回归: {current_weights[0]:.4f}, 分类: {current_weights[1]:.4f}")
    
    metrics = {
        'total_loss': avg_total_loss,
        'regression_loss': avg_reg_loss,
        'classification_loss': avg_cls_loss,
        'side_loss': avg_side_loss,
        'regression_mse': reg_mse,
        'regression_mae': reg_mae,
        'mse_beta2': mse_beta2, 'mse_beta3': mse_beta3,
        'mae_beta2': mae_beta2, 'mae_beta3': mae_beta3,
        'r2_beta2': r2_beta2, 'r2_beta3': r2_beta3,
        'classification_accuracy': val_acc,
        'side_accuracy': side_val_acc,
        'confusion_matrix': cm,
        'side_confusion_matrix': side_cm,
        'reg_predictions': all_reg_preds,
        'reg_labels': all_reg_labels,
        'cls_predictions': all_cls_preds,
        'cls_labels': all_cls_labels,
        'side_predictions': all_side_preds,
        'side_labels': all_side_labels,
        'task_weights': current_weights
    }
    
    return avg_total_loss, metrics


# ===================================================================
# 5. 辅助函数
# ===================================================================

def set_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(config_device):
    """获取计算设备"""
    if config_device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config_device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，回退到CPU")
        device = torch.device('cpu')
    return device


def create_multitask_optimizer_and_criterion_gradnorm(model, gradnorm_trainer, config):
    """创建多任务优化器和损失函数（GradNorm版本）"""
    # 回归损失函数
    loss_type = config['multitask_mlp_config'].get('regression_loss_type', 'mse')
    if loss_type == 'mse':
        regression_criterion = nn.MSELoss()
    elif loss_type == 'mae':
        regression_criterion = nn.L1Loss()
    elif loss_type == 'huber':
        regression_criterion = nn.SmoothL1Loss()
    else:
        regression_criterion = nn.MSELoss()
    
    # 分类损失函数
    classification_criterion = nn.CrossEntropyLoss()
    side_criterion = nn.CrossEntropyLoss()
    
    # 主模型优化器
    optimizer_params = setup_discriminative_lr_multitask(model, config)
    lr = config['learning_rates']['base']
    optimizer = optim.Adam(optimizer_params, lr=lr, weight_decay=config['weight_decay'])
    
    # GradNorm权重优化器
    gradnorm_lr = config.get('gradnorm', {}).get('lr', 0.025)
    gradnorm_optimizer = optim.Adam(gradnorm_trainer.task_weights.parameters(), lr=gradnorm_lr)
    
    return optimizer, gradnorm_optimizer, regression_criterion, classification_criterion, side_criterion


def plot_multitask_results_gradnorm(reg_predictions, reg_labels, cls_predictions, cls_labels, 
                                   side_predictions, side_labels, output_path, epoch, val_metrics, 
                                   weight_history=None, use_side_classification=False):
    """绘制多任务结果（GradNorm版本，支持左右分类）"""
    if use_side_classification:
        fig, axes = plt.subplots(3, 4, figsize=(24, 18))
    else:
        fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    
    # 第一行：回归结果
    # Beta2 预测 vs 真实值
    axes[0, 0].scatter(reg_labels[:, 0], reg_predictions[:, 0], alpha=0.6)
    axes[0, 0].plot([reg_labels[:, 0].min(), reg_labels[:, 0].max()], 
                    [reg_labels[:, 0].min(), reg_labels[:, 0].max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('True Beta2')
    axes[0, 0].set_ylabel('Predicted Beta2')
    axes[0, 0].set_title('Beta2 Predictions vs True Values')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].text(0.05, 0.95, f'MSE: {val_metrics["mse_beta2"]:.6f}\nR²: {val_metrics["r2_beta2"]:.6f}', 
                   transform=axes[0, 0].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Beta3 预测 vs 真实值
    axes[0, 1].scatter(reg_labels[:, 1], reg_predictions[:, 1], alpha=0.6)
    axes[0, 1].plot([reg_labels[:, 1].min(), reg_labels[:, 1].max()], 
                    [reg_labels[:, 1].min(), reg_labels[:, 1].max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('True Beta3')
    axes[0, 1].set_ylabel('Predicted Beta3')
    axes[0, 1].set_title('Beta3 Predictions vs True Values')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].text(0.05, 0.95, f'MSE: {val_metrics["mse_beta3"]:.6f}\nR²: {val_metrics["r2_beta3"]:.6f}', 
                   transform=axes[0, 1].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2D散点图：(beta2, beta3)空间
    axes[0, 2].scatter(reg_labels[:, 0], reg_labels[:, 1], alpha=0.6, label='True', s=30)
    axes[0, 2].scatter(reg_predictions[:, 0], reg_predictions[:, 1], alpha=0.6, label='Predicted', s=30)
    axes[0, 2].set_xlabel('Beta2')
    axes[0, 2].set_ylabel('Beta3')
    axes[0, 2].set_title('Predictions in (Beta2, Beta3) Space')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 任务权重变化图
    if weight_history is not None and len(weight_history) > 0:
        steps = np.arange(len(weight_history))
        axes[0, 3].plot(steps, weight_history[:, 0], label='Regression Weight', linewidth=2)
        axes[0, 3].plot(steps, weight_history[:, 1], label='Classification Weight', linewidth=2)
        if use_side_classification and weight_history.shape[1] > 2:
            axes[0, 3].plot(steps, weight_history[:, 2], label='Side Classification Weight', linewidth=2)
        axes[0, 3].set_xlabel('Training Steps')
        axes[0, 3].set_ylabel('Task Weight')
        axes[0, 3].set_title('GradNorm Task Weight Evolution')
        axes[0, 3].legend()
        axes[0, 3].grid(True, alpha=0.3)
    else:
        # 显示当前权重信息
        current_weights = val_metrics.get('task_weights', [1.0, 1.0])
        weight_text = f'Current Task Weights:\n\nRegression: {current_weights[0]:.4f}\nClassification: {current_weights[1]:.4f}'
        if use_side_classification and len(current_weights) > 2:
            weight_text += f'\nSide Classification: {current_weights[2]:.4f}'
        axes[0, 3].text(0.5, 0.5, weight_text, 
                       transform=axes[0, 3].transAxes, fontsize=14,
                       horizontalalignment='center', verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        axes[0, 3].set_title('GradNorm Task Weights')
        axes[0, 3].axis('off')
    
    # 第二行：类别分类结果
    cm = val_metrics['confusion_matrix']
    class_names = ['Class 0', 'Class 1', 'Class 2']
    
    # 混淆矩阵
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[1, 0])
    axes[1, 0].set_title('Classification Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('True')
    
    # 类别分布对比
    true_counts = np.bincount(cls_labels, minlength=3)
    pred_counts = np.bincount(cls_predictions, minlength=3)
    x = np.arange(3)
    width = 0.35
    
    axes[1, 1].bar(x - width/2, true_counts, width, label='True', alpha=0.7)
    axes[1, 1].bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.7)
    axes[1, 1].set_xlabel('Class')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Class Distribution Comparison')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(class_names)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 准确率信息
    summary_text = f'Classification Accuracy: {val_metrics["classification_accuracy"]:.2f}%\n\n'
    summary_text += f'Regression MSE: {val_metrics["regression_mse"]:.6f}\n'
    summary_text += f'Regression MAE: {val_metrics["regression_mae"]:.6f}\n\n'
    summary_text += f'Total Loss: {val_metrics["total_loss"]:.4f}\n'
    summary_text += f'Regression Loss: {val_metrics["regression_loss"]:.4f}\n'
    summary_text += f'Classification Loss: {val_metrics["classification_loss"]:.4f}'
    if use_side_classification:
        summary_text += f'\nSide Loss: {val_metrics.get("side_loss", 0):.4f}'
    
    axes[1, 2].text(0.5, 0.5, summary_text, 
                   transform=axes[1, 2].transAxes, fontsize=11,
                   horizontalalignment='center', verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[1, 2].set_title('Multi-Task Performance Summary')
    axes[1, 2].axis('off')
    
    # GradNorm信息
    gradnorm_info = f'GradNorm Configuration:\n\n'
    if 'gradnorm_loss' in val_metrics:
        gradnorm_info += f'GradNorm Loss: {val_metrics["gradnorm_loss"]:.4f}\n'
    current_weights = val_metrics.get('task_weights', [1.0, 1.0])
    if len(current_weights) >= 2:
        gradnorm_info += f'Weight Ratio (Reg/Cls): {current_weights[0]/current_weights[1]:.3f}\n'
    if use_side_classification and len(current_weights) > 2:
        gradnorm_info += f'Weight Ratio (Reg/Side): {current_weights[0]/current_weights[2]:.3f}\n'
    gradnorm_info += '\nTask Balancing:\nAutomatically adjusted\nvia gradient norms'
    
    axes[1, 3].text(0.5, 0.5, gradnorm_info, 
                   transform=axes[1, 3].transAxes, fontsize=11,
                   horizontalalignment='center', verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[1, 3].set_title('GradNorm Status')
    axes[1, 3].axis('off')
    
    # 第三行：左右分类结果（如果启用）
    if use_side_classification and len(side_predictions) > 0:
        side_cm = val_metrics.get('side_confusion_matrix')
        if side_cm is not None:
            side_class_names = ['Left', 'Right']
            
            # 左右分类混淆矩阵
            sns.heatmap(side_cm, annot=True, fmt='d', cmap='Greens', 
                        xticklabels=side_class_names, yticklabels=side_class_names, ax=axes[2, 0])
            axes[2, 0].set_title('Side Classification Confusion Matrix')
            axes[2, 0].set_xlabel('Predicted')
            axes[2, 0].set_ylabel('True')
            
            # 左右分布对比
            side_true_counts = np.bincount(side_labels, minlength=2)
            side_pred_counts = np.bincount(side_predictions, minlength=2)
            x_side = np.arange(2)
            
            axes[2, 1].bar(x_side - width/2, side_true_counts, width, label='True', alpha=0.7)
            axes[2, 1].bar(x_side + width/2, side_pred_counts, width, label='Predicted', alpha=0.7)
            axes[2, 1].set_xlabel('Side')
            axes[2, 1].set_ylabel('Count')
            axes[2, 1].set_title('Side Distribution Comparison')
            axes[2, 1].set_xticks(x_side)
            axes[2, 1].set_xticklabels(side_class_names)
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)
            
            # 左右分类准确率
            side_acc_text = f'Side Classification\n\nAccuracy: {val_metrics.get("side_accuracy", 0):.2f}%\n\n'
            side_acc_text += f'Total Samples: {len(side_labels)}\n'
            side_acc_text += f'Left Samples: {side_true_counts[0]}\n'
            side_acc_text += f'Right Samples: {side_true_counts[1]}'
            
            axes[2, 2].text(0.5, 0.5, side_acc_text, 
                           transform=axes[2, 2].transAxes, fontsize=12,
                           horizontalalignment='center', verticalalignment='center',
                           bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
            axes[2, 2].set_title('Side Classification Performance')
            axes[2, 2].axis('off')
            
            # 对称分割说明
            split_info = 'Symmetric Split:\n\n'
            split_info += 'Each image is split along\nthe center axis and\nmirrored to create\n'
            split_info += 'left and right versions.\n\n'
            split_info += 'This augmentation increases\ndata diversity and helps\n'
            split_info += 'the model learn\nleft-right features.'
            
            axes[2, 3].text(0.5, 0.5, split_info, 
                           transform=axes[2, 3].transAxes, fontsize=11,
                           horizontalalignment='center', verticalalignment='center',
                           bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
            axes[2, 3].set_title('Symmetric Split Info')
            axes[2, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_best_multitask_outputs_gradnorm(reg_predictions, reg_labels, cls_predictions, cls_labels, 
                                        side_predictions, side_labels, output_dir, val_metrics, 
                                        weight_history=None, use_side_classification=False):
    """保存最佳多任务模型的输出数据和图片（GradNorm版本，支持左右分类）"""
    # 保存图片
    plot_path = output_dir / 'best_multitask_gradnorm_output.png'
    plot_multitask_results_gradnorm(
        reg_predictions, 
        reg_labels, 
        cls_predictions, 
        cls_labels,
        side_predictions,
        side_labels,
        plot_path, 
        epoch=None,
        val_metrics=val_metrics,
        weight_history=weight_history,
        use_side_classification=use_side_classification
    )
    
    # 保存数据到npz文件
    npz_path = output_dir / 'best_multitask_gradnorm_output.npz'
    save_data = {
        'reg_predictions': reg_predictions,
        'reg_labels': reg_labels,
        'cls_predictions': cls_predictions,
        'cls_labels': cls_labels,
    }
    
    # 添加左右分类数据
    if use_side_classification:
        save_data['side_predictions'] = side_predictions
        save_data['side_labels'] = side_labels
    
    # 添加数值型的度量指标
    for k, v in val_metrics.items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            save_data[k] = v
        elif isinstance(v, np.ndarray) and v.ndim <= 2:
            save_data[k] = v
    
    # 保存权重历史
    if weight_history is not None:
        save_data['weight_history'] = weight_history
    
    np.savez(npz_path, **save_data)
    
    print(f"最佳多任务GradNorm模型输出已保存:")
    print(f"  - 图片: {plot_path}")
    print(f"  - 数据: {npz_path}")


# ===================================================================
# 6. 主函数
# ===================================================================

def main():
    """主训练函数（GradNorm版本，支持对称分割）"""
    # 加载配置
    with open('config_sym.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置基本参数
    set_seed(config['seed'])
    device = get_device(config['device'])
    print(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(config['output_dir']) / config['experiment_name']
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / 'logs'; logs_dir.mkdir(exist_ok=True)
    checkpoints_dir = output_dir / 'checkpoints'; checkpoints_dir.mkdir(exist_ok=True)
    plots_dir = output_dir / 'plots'; plots_dir.mkdir(exist_ok=True)
    
    # 设置自定义日志记录器
    log_file = output_dir / 'training.log'
    custom_logger = CustomLogger(log_file)
    sys.stdout = custom_logger
    
    try:
        # 设置Python日志系统
        logging.basicConfig(
            level=getattr(logging, config['logging']['level']),
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.StreamHandler()
            ]
        )
        
        # TensorBoard写入器
        if config['logging']['tensorboard']:
            writer = SummaryWriter(str(logs_dir))
        else:
            writer = None
        
        # 创建数据集
        train_dataset = MultiTaskInterferenceDataset(
            Path(config['data_path']) / 'train', 
            config=config, 
            split='train'
        )
        val_dataset = MultiTaskInterferenceDataset(
            Path(config['data_path']) / 'val', 
            config=config, 
            split='val'
        )
        
        # 创建模型
        model = create_model(config, device=device)
        model.to(device)
        
        # 判断是否启用左右分类
        use_side_classification = config.get('symmetric_split', {}).get('enabled', False)
        num_tasks = 3 if use_side_classification else 2
        
        # 创建GradNorm训练器
        gradnorm_config = config.get('gradnorm', {})
        gradnorm_alpha = gradnorm_config.get('alpha', 1.5)
        gradnorm_trainer = GradNormTrainer(model, num_tasks=num_tasks, alpha=gradnorm_alpha)
        gradnorm_trainer.task_weights.to(device)
        
        # 创建优化器和损失函数
        optimizer, gradnorm_optimizer, regression_criterion, classification_criterion, side_criterion = \
            create_multitask_optimizer_and_criterion_gradnorm(model, gradnorm_trainer, config)
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # 最佳模型跟踪
        best_total_loss = float('inf')
        best_r2_score = -float('inf')
        best_model_path = checkpoints_dir / 'best_multitask_gradnorm_model.pth'
        all_weight_history = []
        
        # 开始训练
        logging.info(f"开始多任务GradNorm训练（对称分割模式）...")
        logging.info(f"对称分割: {'启用' if use_side_classification else '禁用'}")
        logging.info(f"任务数量: {num_tasks}")
        logging.info(f"GradNorm配置:")
        logging.info(f"  - Alpha: {gradnorm_alpha}")
        logging.info(f"  - GradNorm学习率: {gradnorm_config.get('lr', 0.025)}")
        logging.info(f"  - GradNorm权重: {gradnorm_config.get('weight', 0.1)}")
        
        multitask_conf = config['multitask_mlp_config']
        logging.info(f"多任务MLP配置:")
        logging.info(f"  - 特征维度: {multitask_conf['feature_dim']}")
        logging.info(f"  - 共享层: {multitask_conf.get('shared_dims', 'None')}")
        logging.info(f"  - 回归层: {multitask_conf['regression_dims']}")
        logging.info(f"  - 分类层: {multitask_conf['classification_dims']}")
        if use_side_classification:
            logging.info(f"  - 左右分类层: {multitask_conf.get('side_classification_dims', [128, 64])}")
        
        for epoch in range(1, config['epochs'] + 1):
            logging.info(f"\n{'='*50}\nEpoch {epoch}/{config['epochs']}\n{'='*50}")
            
            # 创建数据加载器
            train_loader = DataLoader(
                train_dataset, 
                batch_size=config['batch_size'], 
                shuffle=True, 
                num_workers=config['num_workers']
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=config['batch_size'], 
                shuffle=False, 
                num_workers=config['num_workers']
            )
            
            # 训练和验证
            train_metrics = train_epoch_multitask_gradnorm(
                model, train_loader, regression_criterion, classification_criterion, 
                side_criterion, optimizer, gradnorm_optimizer, gradnorm_trainer, 
                device, epoch, config
            )
            
            val_total_loss, val_metrics = validate_epoch_multitask_gradnorm(
                model, val_loader, regression_criterion, classification_criterion, 
                side_criterion, gradnorm_trainer, device, epoch, config
            )
            
            # 记录权重历史
            all_weight_history.append(train_metrics['weight_history'])
            
            # 学习率调度
            scheduler.step(val_total_loss)
            
            # 记录到TensorBoard
            if writer:
                # 损失
                writer.add_scalar('Loss/Train_Total', train_metrics['total_loss'], epoch)
                writer.add_scalar('Loss/Train_Regression', train_metrics['regression_loss'], epoch)
                writer.add_scalar('Loss/Train_Classification', train_metrics['classification_loss'], epoch)
                if use_side_classification:
                    writer.add_scalar('Loss/Train_Side', train_metrics['side_loss'], epoch)
                writer.add_scalar('Loss/Train_GradNorm', train_metrics['gradnorm_loss'], epoch)
                writer.add_scalar('Loss/Val_Total', val_total_loss, epoch)
                writer.add_scalar('Loss/Val_Regression', val_metrics['regression_loss'], epoch)
                writer.add_scalar('Loss/Val_Classification', val_metrics['classification_loss'], epoch)
                if use_side_classification:
                    writer.add_scalar('Loss/Val_Side', val_metrics['side_loss'], epoch)
                
                # 准确率
                writer.add_scalar('Accuracy/Train', train_metrics['train_accuracy'], epoch)
                writer.add_scalar('Accuracy/Validation', val_metrics['classification_accuracy'], epoch)
                if use_side_classification:
                    writer.add_scalar('Accuracy/Train_Side', train_metrics['side_train_accuracy'], epoch)
                    writer.add_scalar('Accuracy/Validation_Side', val_metrics['side_accuracy'], epoch)
                
                # 回归指标
                writer.add_scalar('Regression/MSE', val_metrics['regression_mse'], epoch)
                writer.add_scalar('Regression/MAE', val_metrics['regression_mae'], epoch)
                writer.add_scalar('Regression/R2_Beta2', val_metrics['r2_beta2'], epoch)
                writer.add_scalar('Regression/R2_Beta3', val_metrics['r2_beta3'], epoch)
                
                # 任务权重
                current_weights = train_metrics['task_weights']
                writer.add_scalar('GradNorm/Weight_Regression', current_weights[0], epoch)
                writer.add_scalar('GradNorm/Weight_Classification', current_weights[1], epoch)
                if use_side_classification and len(current_weights) > 2:
                    writer.add_scalar('GradNorm/Weight_Side', current_weights[2], epoch)
                    writer.add_scalar('GradNorm/Weight_Ratio_Reg_Cls', current_weights[0]/current_weights[1], epoch)
                    writer.add_scalar('GradNorm/Weight_Ratio_Reg_Side', current_weights[0]/current_weights[2], epoch)
                else:
                    writer.add_scalar('GradNorm/Weight_Ratio', current_weights[0]/current_weights[1], epoch)
                
                # 学习率
                writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            # 绘制多任务结果（定期）
            if config['validation']['plot_predictions'] and epoch % config['validation']['plot_interval'] == 0:
                plot_path = plots_dir / f'multitask_gradnorm_results_epoch_{epoch}.png'
                combined_weight_history = np.vstack(all_weight_history) if all_weight_history else None
                plot_multitask_results_gradnorm(
                    val_metrics['reg_predictions'], 
                    val_metrics['reg_labels'],
                    val_metrics['cls_predictions'],
                    val_metrics['cls_labels'],
                    val_metrics['side_predictions'],
                    val_metrics['side_labels'],
                    plot_path, 
                    epoch, 
                    val_metrics,
                    weight_history=combined_weight_history,
                    use_side_classification=use_side_classification
                )
            
            # 计算综合评分（考虑R²分数）
            current_r2 = (0.8 * val_metrics['r2_beta2'] + val_metrics['r2_beta3']) / 2
            
            # 保存最佳模型
            if current_r2 > best_r2_score:
                best_total_loss = val_total_loss
                best_r2_score = current_r2
                logging.info(f"新的最佳模型! 验证总损失: {best_total_loss:.6f}, R²分数: {best_r2_score:.6f}")
                
                if config['save_options']['save_best_only']:
                    save_payload = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'gradnorm_weights_state_dict': gradnorm_trainer.task_weights.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'gradnorm_optimizer_state_dict': gradnorm_optimizer.state_dict(),
                        'best_total_loss': best_total_loss,
                        'best_r2_score': best_r2_score,
                        'val_metrics': val_metrics,
                        'train_metrics': train_metrics,
                        'config': config,
                        'gradnorm_trainer_state': {
                            'initial_losses': gradnorm_trainer.initial_losses,
                            'running_losses': gradnorm_trainer.running_losses,
                            'alpha': gradnorm_trainer.alpha
                        }
                    }
                    torch.save(save_payload, best_model_path)
                    
                    # 保存最佳模型的输出
                    combined_weight_history = np.vstack(all_weight_history) if all_weight_history else None
                    save_best_multitask_outputs_gradnorm(
                        val_metrics['reg_predictions'], 
                        val_metrics['reg_labels'],
                        val_metrics['cls_predictions'],
                        val_metrics['cls_labels'],
                        val_metrics['side_predictions'],
                        val_metrics['side_labels'],
                        output_dir, 
                        val_metrics,
                        weight_history=combined_weight_history,
                        use_side_classification=use_side_classification
                    )
            
            # 输出训练日志
            logging.info(f"Epoch {epoch} 总结:")
            log_text = f"  训练 - 总损失: {train_metrics['total_loss']:.4f}, 回归: {train_metrics['regression_loss']:.4f}, "
            log_text += f"分类: {train_metrics['classification_loss']:.4f}"
            if use_side_classification:
                log_text += f", 左右: {train_metrics['side_loss']:.4f}"
            log_text += f", GradNorm: {train_metrics['gradnorm_loss']:.4f}"
            logging.info(log_text)
            
            logging.info(f"  训练准确率: {train_metrics['train_accuracy']:.2f}%")
            if use_side_classification:
                logging.info(f"  训练左右准确率: {train_metrics['side_train_accuracy']:.2f}%")
            
            val_log = f"  验证 - 总损失: {val_total_loss:.4f}, 回归MSE: {val_metrics['regression_mse']:.6f}, "
            val_log += f"分类准确率: {val_metrics['classification_accuracy']:.2f}%"
            if use_side_classification:
                val_log += f", 左右准确率: {val_metrics['side_accuracy']:.2f}%"
            logging.info(val_log)
            
            current_weights = val_metrics['task_weights']
            if use_side_classification and len(current_weights) > 2:
                logging.info(f"  当前任务权重 - 回归: {current_weights[0]:.4f}, 分类: {current_weights[1]:.4f}, 左右: {current_weights[2]:.4f}")
            else:
                logging.info(f"  当前任务权重 - 回归: {current_weights[0]:.4f}, 分类: {current_weights[1]:.4f}")
        
        if writer:
            writer.close()
        
        logging.info(f"\n{'='*50}")
        logging.info(f"多任务GradNorm训练完成!")
        logging.info(f"最佳验证总损失: {best_total_loss:.6f}")
        logging.info(f"最佳R²分数: {best_r2_score:.6f}")
        logging.info(f"模型保存路径: {best_model_path}")
        logging.info(f"{'='*50}")
        
    finally:
        # 确保恢复标准输出并关闭日志文件
        sys.stdout = custom_logger.terminal
        custom_logger.close()


if __name__ == '__main__':
    main()