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
        # 使用nn.Parameter使其成为模型的可学习参数
        self.weights = nn.Parameter(torch.ones(num_tasks))
        self.num_tasks = num_tasks

    def forward(self):
        # 在前向传播时，返回当前的权重
        return self.weights
    
    def renormalize(self):
        """
        在权重更新后对其进行重新规范化，以防止权重漂移到0或无穷大。
        这确保了权重的相对比例得以保持，同时总和保持不变。
        """
        # F.softmax会确保权重为正且和为1，乘以任务数量可以使它们的平均值保持在1左右
        renorm_weights = self.num_tasks * F.softmax(self.weights, dim=0)
        # 使用.data直接修改权重值，而不会影响梯度计算图
        self.weights.data = renorm_weights.data


class GradNormTrainer:
    """GradNorm训练器，负责管理多任务权重的动态调整"""
    def __init__(self, model, num_tasks, alpha=1.5):
        """
        初始化GradNorm训练器
        
        Args:
            model: 多任务模型
            num_tasks: 任务数量
            alpha: GradNorm超参数，控制任务平衡的强度
        """
        self.model = model
        self.num_tasks = num_tasks
        self.alpha = alpha
        
        # 创建任务权重模块
        self.task_weights = GradNormWeights(num_tasks)
        
        # 获取最后一个共享层的参数，用于计算梯度
        self.shared_params = self._get_shared_parameters()
        
        # 初始化任务损失的运行平均
        self.initial_losses = None
        self.running_losses = None
        
    def _get_shared_parameters(self):
        """获取用于计算梯度的共享参数"""
        # 获取特征提取器的最后一层参数
        # 这里我们使用特征投射层的最后一个线性层
        for name, module in self.model.feature_extractor.feature_proj.named_modules():
            if isinstance(module, nn.Linear):
                last_linear = module
        
        # 返回最后一个线性层的权重参数
        return last_linear.weight
    
    def compute_grad_norm(self, losses):
        """
        计算并应用GradNorm算法
        
        Args:
            losses: 各任务的损失张量 [task1_loss, task2_loss, ...]
        
        Returns:
            weighted_loss: 加权后的总损失
        """
        # 如果是第一次运行，初始化损失记录
        if self.initial_losses is None:
            self.initial_losses = losses.clone().detach()
            self.running_losses = losses.clone().detach()
        
        # 更新运行平均损失
        decay = 0.1  # 指数移动平均的衰减率
        self.running_losses = (1 - decay) * self.running_losses + decay * losses.detach()
        
        # 获取当前任务权重
        weights = self.task_weights()
        
        # 计算加权损失
        weighted_loss = torch.sum(weights * losses)
        
        # 清零梯度
        if self.shared_params.grad is not None:
            self.shared_params.grad.zero_()
        
        # 分别计算每个任务对共享参数的梯度
        grad_norms = []
        for i, loss in enumerate(losses):
            # 计算当前任务对共享参数的梯度
            task_grad = torch.autograd.grad(
                weights[i] * loss, 
                self.shared_params, 
                retain_graph=True,
                create_graph=True
            )[0]
            
            # 计算梯度的L2范数
            grad_norm = torch.norm(task_grad)
            grad_norms.append(grad_norm)
        
        grad_norms = torch.stack(grad_norms)
        
        # 计算相对损失率
        loss_ratios = self.running_losses / self.initial_losses
        
        # 计算平均梯度范数
        mean_grad_norm = torch.mean(grad_norms.detach())
        
        # 计算目标梯度范数
        # r_i = loss_ratio_i^alpha / mean(loss_ratio^alpha)
        relative_rates = torch.pow(loss_ratios, self.alpha)
        mean_relative_rate = torch.mean(relative_rates)
        target_grad_norms = mean_grad_norm * (relative_rates / mean_relative_rate)
        
        # 计算GradNorm损失
        gradnorm_loss = torch.sum(torch.abs(grad_norms - target_grad_norms.detach()))
        
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
            
        # 对数变换
        tensor = torch.log1p(tensor + self.epsilon)
        
        # 归一化到 [0, 1]
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
# 2. 数据集类（修改为单物理量回归+分类）
# ===================================================================

class MultiTaskInterferenceDataset(Dataset):
    """
    多任务干涉数据集类
    - 同时进行回归任务（预测beta3）和分类任务（预测类别）
    """
    def __init__(self, data_path, config, split='train'):
        self.data_path = Path(data_path)
        self.config = config
        self.split = split
        self.mode = config['inference_mode']
        
        # 初始化预处理器
        self.preprocessor = DataPreprocessor(config)
        
        # 加载数据文件
        self.file_paths = sorted(list(self.data_path.glob('**/*.npy')))
        if not self.file_paths:
            raise ValueError(f"在路径 {self.data_path} 中没有找到任何.npy文件!")
            
        # 解析标签（包括回归和分类）
        self.regression_labels, self.classification_labels, self.file_paths = self._parse_labels()
        
        print(f"'{self.split}' 数据集初始化完成:")
        print(f"  - 文件数量: {len(self.file_paths)}")
        print(f"  - 推理模式: {self.mode}")
        print(f"  - 类别数量: 3")

    def _parse_labels(self):
        """解析beta3值标签和分类标签"""
        beta_pattern = re.compile(r"beta3_([\d.]+)")
        class_pattern = re.compile(r"class(\d)")
        
        beta_values = []
        class_labels = []
        valid_file_paths = []
        
        for file_path in self.file_paths:
            beta_match = beta_pattern.search(file_path.stem)
            class_match = class_pattern.search(file_path.stem)
            
            if beta_match and class_match:
                try:
                    beta3 = float(beta_match.group(1))
                    class_id = int(class_match.group(1))
                    
                    if class_id in [0, 1, 2]:
                        beta_values.append(beta3)
                        class_labels.append(class_id)
                        valid_file_paths.append(file_path)
                    else:
                        print(f"警告: 文件 {file_path.name} 包含无效的类别ID {class_id}, 已跳过。")
                except (ValueError, IndexError):
                    print(f"警告: 解析文件 {file_path.name} 失败, 已跳过。")
            else:
                print(f"警告: 文件 {file_path.name} 无法匹配beta3或类别标签, 已跳过。")

        if not beta_values or not class_labels:
            raise ValueError(f"在路径 {self.data_path} 中没有找到任何有效的数据文件!")

        regression_labels = [torch.tensor([b], dtype=torch.float32) for b in beta_values]
        
        print(f"多任务模式: 加载 {len(regression_labels)} 个样本")
        print(f"  - 回归标签: beta3值")
        print(f"  - 分类标签分布: {dict(sorted(zip(*np.unique(class_labels, return_counts=True))))}")
        
        return regression_labels, class_labels, valid_file_paths

    def __len__(self):
        return len(self.regression_labels)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image_data = np.load(file_path)
        
        image_tensor = torch.from_numpy(image_data.astype(np.float32)).unsqueeze(0)
        
        image_tensor = self.preprocessor(image_tensor)
        
        regression_label = self.regression_labels[idx]
        classification_label = self.classification_labels[idx]
        
        return image_tensor, regression_label, classification_label


# ===================================================================
# 3. 多任务模型定义
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
    """ResNet + 多任务MLP模型（回归beta3 + 分类）"""
    def __init__(self, config):
        super().__init__()
        mlp_config = config['multitask_mlp_config']
        model_config = config.get('model_config', {})
        
        feature_dim = mlp_config['feature_dim']
        dropout_rate = model_config.get('dropout_rate', 0.1)
        self.feature_extractor = ResNetFeatureExtractor(feature_dim, dropout_rate)
        
        self.shared_layers = self._build_shared_layers(mlp_config, feature_dim)
        
        shared_output_dim = mlp_config['shared_dims'][-1] if mlp_config.get('shared_dims') else feature_dim
        self.regression_head = self._build_regression_head(mlp_config, shared_output_dim)
        self.classification_head = self._build_classification_head(mlp_config, shared_output_dim)
        
        self._initialize_weights()
    
    def _build_shared_layers(self, mlp_config, input_dim):
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
        
        layers.append(nn.Linear(current_dim, 1))
        
        return nn.Sequential(*layers)
    
    def _build_classification_head(self, mlp_config, input_dim):
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
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        
        shared_features = self.shared_layers(features)
        
        regression_output = self.regression_head(shared_features)
        classification_output = self.classification_head(shared_features)
        
        regression_output = F.softplus(regression_output) + 1e-6
        
        return regression_output, classification_output


def create_model(config, device='cpu'):
    mode = config['inference_mode']
    
    if mode == 'multitask_mlp':
        return MultiTaskResNetMLP(config)
    else:
        raise ValueError(f"当前只支持 'multitask_mlp' 模式，收到: {mode}")


# ===================================================================
# 4. 多任务训练与验证函数（集成GradNorm）
# ===================================================================

def setup_discriminative_lr_multitask(model, config):
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
                                  optimizer, gradnorm_optimizer, gradnorm_trainer, device, epoch, config):
    """使用GradNorm的多任务训练一个epoch"""
    model.train()
    gradnorm_trainer.task_weights.train()
    
    running_total_loss = 0.0
    running_reg_loss = 0.0
    running_cls_loss = 0.0
    running_gradnorm_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    weight_history = []
    
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for i, (inputs, reg_labels, cls_labels) in enumerate(progress_bar):
        inputs = inputs.to(device)
        reg_labels = reg_labels.to(device)
        cls_labels = cls_labels.to(device)
        
        optimizer.zero_grad()
        gradnorm_optimizer.zero_grad()
        
        reg_outputs, cls_outputs = model(inputs)
        
        reg_loss = regression_criterion(reg_outputs, reg_labels)
        cls_loss = classification_criterion(cls_outputs, cls_labels)
        task_losses = torch.stack([reg_loss, cls_loss])
        
        weighted_loss, gradnorm_loss = gradnorm_trainer.compute_grad_norm(task_losses)
        
        gradnorm_weight = config.get('gradnorm', {}).get('weight', 0.1)
        total_loss = weighted_loss + gradnorm_weight * gradnorm_loss
        
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(gradnorm_trainer.task_weights.parameters(), max_norm=1.0)
        
        optimizer.step()
        gradnorm_optimizer.step()
        
        gradnorm_trainer.task_weights.renormalize()
        
        running_total_loss += total_loss.item()
        running_reg_loss += reg_loss.item()
        running_cls_loss += cls_loss.item()
        running_gradnorm_loss += gradnorm_loss.item()
        
        _, predicted = cls_outputs.max(1)
        correct_predictions += predicted.eq(cls_labels).sum().item()
        total_samples += cls_labels.size(0)
        
        current_weights = gradnorm_trainer.get_current_weights()
        weight_history.append(current_weights)
        
        progress_bar.set_postfix(
            loss=f"{total_loss.item():.4f}", 
            acc=f"{100.*correct_predictions/total_samples:.2f}%",
            w_reg=f"{current_weights[0]:.3f}",
            w_cls=f"{current_weights[1]:.3f}"
        )
    
    avg_total_loss = running_total_loss / len(train_loader)
    avg_reg_loss = running_reg_loss / len(train_loader)
    avg_cls_loss = running_cls_loss / len(train_loader)
    avg_gradnorm_loss = running_gradnorm_loss / len(train_loader)
    train_acc = 100. * correct_predictions / total_samples if total_samples > 0 else 0.0
    
    final_weights = gradnorm_trainer.get_current_weights()
    
    metrics = {
        'total_loss': avg_total_loss,
        'regression_loss': avg_reg_loss,
        'classification_loss': avg_cls_loss,
        'gradnorm_loss': avg_gradnorm_loss,
        'train_accuracy': train_acc,
        'task_weights': final_weights,
        'weight_history': np.array(weight_history)
    }
    
    return metrics


def validate_epoch_multitask_gradnorm(model, val_loader, regression_criterion, classification_criterion, 
                                     gradnorm_trainer, device, epoch):
    """使用GradNorm权重的多任务验证一个epoch"""
    model.eval()
    gradnorm_trainer.task_weights.eval()
    
    running_total_loss = 0.0
    running_reg_loss = 0.0
    running_cls_loss = 0.0
    
    all_reg_preds, all_reg_labels = [], []
    all_cls_preds, all_cls_labels = [], []
    
    current_weights = gradnorm_trainer.get_current_weights()
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
        for inputs, reg_labels, cls_labels in progress_bar:
            inputs = inputs.to(device)
            reg_labels = reg_labels.to(device)
            cls_labels = cls_labels.to(device)
            
            reg_outputs, cls_outputs = model(inputs)
            
            reg_loss = regression_criterion(reg_outputs, reg_labels)
            cls_loss = classification_criterion(cls_outputs, cls_labels)
            
            total_loss = current_weights[0] * reg_loss + current_weights[1] * cls_loss
            
            running_total_loss += total_loss.item()
            running_reg_loss += reg_loss.item()
            running_cls_loss += cls_loss.item()
            
            all_reg_preds.extend(reg_outputs.cpu().numpy())
            all_reg_labels.extend(reg_labels.cpu().numpy())
            
            _, predicted = cls_outputs.max(1)
            all_cls_preds.extend(predicted.cpu().numpy())
            all_cls_labels.extend(cls_labels.cpu().numpy())
    
    avg_total_loss = running_total_loss / len(val_loader)
    avg_reg_loss = running_reg_loss / len(val_loader)
    avg_cls_loss = running_cls_loss / len(val_loader)
    
    all_reg_preds = np.array(all_reg_preds).flatten()
    all_reg_labels = np.array(all_reg_labels).flatten()
    
    reg_mse = mean_squared_error(all_reg_labels, all_reg_preds)
    reg_mae = mean_absolute_error(all_reg_labels, all_reg_preds)
    r2_beta3 = r2_score(all_reg_labels, all_reg_preds)
    
    val_acc = 100. * np.sum(np.array(all_cls_preds) == np.array(all_cls_labels)) / len(all_cls_labels)
    cm = confusion_matrix(all_cls_labels, all_cls_preds)
    
    metrics = {
        'total_loss': avg_total_loss,
        'regression_loss': avg_reg_loss,
        'classification_loss': avg_cls_loss,
        'regression_mse': reg_mse,
        'regression_mae': reg_mae,
        'r2_beta3': r2_beta3,
        'classification_accuracy': val_acc,
        'confusion_matrix': cm,
        'reg_predictions': all_reg_preds,
        'reg_labels': all_reg_labels,
        'cls_predictions': all_cls_preds,
        'cls_labels': all_cls_labels,
        'task_weights': current_weights
    }
    
    return avg_total_loss, metrics


# ===================================================================
# 5. 辅助函数
# ===================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(config_device):
    if config_device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config_device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，回退到CPU")
        device = torch.device('cpu')
    return device


def create_multitask_optimizer_and_criterion_gradnorm(model, gradnorm_trainer, config):
    loss_type = config['multitask_mlp_config'].get('regression_loss_type', 'mse')
    if loss_type == 'mse':
        regression_criterion = nn.MSELoss()
    elif loss_type == 'mae':
        regression_criterion = nn.L1Loss()
    elif loss_type == 'huber':
        regression_criterion = nn.SmoothL1Loss()
    else:
        regression_criterion = nn.MSELoss()
    
    classification_criterion = nn.CrossEntropyLoss()
    
    optimizer_params = setup_discriminative_lr_multitask(model, config)
    lr = config['learning_rates']['base']
    optimizer = optim.Adam(optimizer_params, lr=lr, weight_decay=config['weight_decay'])
    
    gradnorm_lr = config.get('gradnorm', {}).get('lr', 0.025)
    gradnorm_optimizer = optim.Adam(gradnorm_trainer.task_weights.parameters(), lr=gradnorm_lr)
    
    return optimizer, gradnorm_optimizer, regression_criterion, classification_criterion


def plot_multitask_results_gradnorm(reg_predictions, reg_labels, cls_predictions, cls_labels, 
                                   output_path, epoch, val_metrics, weight_history=None):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    axes[0, 0].scatter(reg_labels, reg_predictions, alpha=0.6)
    axes[0, 0].plot([reg_labels.min(), reg_labels.max()], 
                    [reg_labels.min(), reg_labels.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('True beta3')
    axes[0, 0].set_ylabel('Predicted beta3')
    axes[0, 0].set_title('beta3 Predictions vs True Values')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].text(0.05, 0.95, f'MSE: {val_metrics["regression_mse"]:.6f}\nR²: {val_metrics["r2_beta3"]:.6f}', 
                   transform=axes[0, 0].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    residuals = reg_predictions - reg_labels
    axes[0, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Residuals (Predicted - True)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('beta3 Residuals Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].text(0.05, 0.95, f'MAE: {val_metrics["regression_mae"]:.6f}', 
                   transform=axes[0, 1].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if weight_history is not None and len(weight_history) > 0:
        steps = np.arange(len(weight_history))
        axes[0, 2].plot(steps, weight_history[:, 0], label='Regression Weight', linewidth=2)
        axes[0, 2].plot(steps, weight_history[:, 1], label='Classification Weight', linewidth=2)
        axes[0, 2].set_xlabel('Training Steps')
        axes[0, 2].set_ylabel('Task Weight')
        axes[0, 2].set_title('GradNorm Task Weight Evolution')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    else:
        current_weights = val_metrics.get('task_weights', [1.0, 1.0])
        axes[0, 2].text(0.5, 0.5, f'Current Task Weights:\n\nRegression: {current_weights[0]:.4f}\nClassification: {current_weights[1]:.4f}', 
                       transform=axes[0, 2].transAxes, fontsize=14,
                       horizontalalignment='center', verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        axes[0, 2].set_title('GradNorm Task Weights')
        axes[0, 2].axis('off')
    
    cm = val_metrics['confusion_matrix']
    class_names = ['Class 0', 'Class 1', 'Class 2']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[1, 0])
    axes[1, 0].set_title('Classification Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('True')
    
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
    
    axes[1, 2].text(0.5, 0.5, f'Multi-Task Performance:\n\n'
                               f'Classification Accuracy: {val_metrics["classification_accuracy"]:.2f}%\n'
                               f'Regression MSE: {val_metrics["regression_mse"]:.6f}\n'
                               f'Regression MAE: {val_metrics["regression_mae"]:.6f}\n'
                               f'Regression R²: {val_metrics["r2_beta3"]:.6f}\n\n'
                               f'Total Loss: {val_metrics["total_loss"]:.4f}\n'
                               f'Regression Loss: {val_metrics["regression_loss"]:.4f}\n'
                               f'Classification Loss: {val_metrics["classification_loss"]:.4f}', 
                   transform=axes[1, 2].transAxes, fontsize=11,
                   horizontalalignment='center', verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[1, 2].set_title('Performance Summary')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_best_multitask_outputs_gradnorm(reg_predictions, reg_labels, cls_predictions, cls_labels, 
                                        output_dir, val_metrics, weight_history=None):
    plot_path = output_dir / 'best_multitask_gradnorm_output.png'
    plot_multitask_results_gradnorm(
        reg_predictions, 
        reg_labels, 
        cls_predictions, 
        cls_labels,
        plot_path, 
        epoch=None,
        val_metrics=val_metrics,
        weight_history=weight_history
    )
    
    npz_path = output_dir / 'best_multitask_gradnorm_output.npz'
    save_data = {
        'reg_predictions': reg_predictions,
        'reg_labels': reg_labels,
        'cls_predictions': cls_predictions,
        'cls_labels': cls_labels,
    }
    
    for k, v in val_metrics.items():
        if isinstance(v, (int, float, np.integer, np.floating)):
            save_data[k] = v
        elif isinstance(v, np.ndarray) and v.ndim <= 2:
            save_data[k] = v
    
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
    """主训练函数（GradNorm版本）"""
    with open('config2.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    set_seed(config['seed'])
    device = get_device(config['device'])
    print(f"使用设备: {device}")
    
    output_dir = Path(config['output_dir']) / config['experiment_name']
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / 'logs'; logs_dir.mkdir(exist_ok=True)
    checkpoints_dir = output_dir / 'checkpoints'; checkpoints_dir.mkdir(exist_ok=True)
    plots_dir = output_dir / 'plots'; plots_dir.mkdir(exist_ok=True)
    
    log_file = output_dir / 'training.log'
    custom_logger = CustomLogger(log_file)
    sys.stdout = custom_logger
    
    try:
        logging.basicConfig(
            level=getattr(logging, config['logging']['level'].upper()),
            format='%(asctime)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        
        writer = SummaryWriter(str(logs_dir)) if config['logging']['tensorboard'] else None
        
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
        
        model = create_model(config, device=device)
        model.to(device)
        
        gradnorm_config = config.get('gradnorm', {})
        gradnorm_alpha = gradnorm_config.get('alpha', 1.5)
        gradnorm_trainer = GradNormTrainer(model, num_tasks=2, alpha=gradnorm_alpha)
        gradnorm_trainer.task_weights.to(device)
        
        optimizer, gradnorm_optimizer, regression_criterion, classification_criterion = \
            create_multitask_optimizer_and_criterion_gradnorm(model, gradnorm_trainer, config)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        best_total_loss = float('inf')
        best_r2_score = -float('inf')
        best_model_path = checkpoints_dir / 'best_multitask_gradnorm_model.pth'
        all_weight_history = []
        
        logging.info("开始多任务GradNorm训练...")
        logging.info(f"GradNorm配置: Alpha={gradnorm_alpha}, LR={gradnorm_config.get('lr', 0.025)}, Weight={gradnorm_config.get('weight', 0.1)}")
        
        multitask_conf = config['multitask_mlp_config']
        logging.info(f"多任务MLP配置: FeatureDim={multitask_conf['feature_dim']}, Shared={multitask_conf.get('shared_dims', 'None')}, "
                     f"Reg={multitask_conf['regression_dims']}, Cls={multitask_conf['classification_dims']}")
        
        for epoch in range(1, config['epochs'] + 1):
            logging.info(f"\n{'='*50}\nEpoch {epoch}/{config['epochs']}\n{'='*50}")
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=config['batch_size'], 
                shuffle=True, 
                num_workers=config['num_workers'],
                pin_memory=True
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=config['batch_size'], 
                shuffle=False, 
                num_workers=config['num_workers'],
                pin_memory=True
            )
            
            train_metrics = train_epoch_multitask_gradnorm(
                model, train_loader, regression_criterion, classification_criterion, 
                optimizer, gradnorm_optimizer, gradnorm_trainer, device, epoch, config
            )
            
            val_total_loss, val_metrics = validate_epoch_multitask_gradnorm(
                model, val_loader, regression_criterion, classification_criterion, 
                gradnorm_trainer, device, epoch
            )
            
            all_weight_history.extend(train_metrics['weight_history'])
            
            scheduler.step(val_total_loss)
            
            if writer:
                writer.add_scalar('Loss/Train_Total', train_metrics['total_loss'], epoch)
                writer.add_scalar('Loss/Train_Regression', train_metrics['regression_loss'], epoch)
                writer.add_scalar('Loss/Train_Classification', train_metrics['classification_loss'], epoch)
                writer.add_scalar('Loss/Train_GradNorm', train_metrics['gradnorm_loss'], epoch)
                writer.add_scalar('Loss/Val_Total', val_total_loss, epoch)
                writer.add_scalar('Loss/Val_Regression', val_metrics['regression_loss'], epoch)
                writer.add_scalar('Loss/Val_Classification', val_metrics['classification_loss'], epoch)
                
                writer.add_scalar('Accuracy/Train', train_metrics['train_accuracy'], epoch)
                writer.add_scalar('Accuracy/Validation', val_metrics['classification_accuracy'], epoch)
                
                writer.add_scalar('Regression/MSE', val_metrics['regression_mse'], epoch)
                writer.add_scalar('Regression/MAE', val_metrics['regression_mae'], epoch)
                writer.add_scalar('Regression/R2_beta3', val_metrics['r2_beta3'], epoch)
                
                current_weights = train_metrics['task_weights']
                writer.add_scalar('GradNorm/Weight_Regression', current_weights[0], epoch)
                writer.add_scalar('GradNorm/Weight_Classification', current_weights[1], epoch)
                if current_weights[1] > 1e-8:
                    writer.add_scalar('GradNorm/Weight_Ratio', current_weights[0]/current_weights[1], epoch)
                
                writer.add_scalar('Learning_Rate', optimizer.param_groups[-1]['lr'], epoch)
            
            if config['validation']['plot_predictions'] and epoch % config['validation']['plot_interval'] == 0:
                plot_path = plots_dir / f'multitask_gradnorm_results_epoch_{epoch}.png'
                combined_weight_history = np.vstack(all_weight_history) if all_weight_history else None
                plot_multitask_results_gradnorm(
                    val_metrics['reg_predictions'], val_metrics['reg_labels'],
                    val_metrics['cls_predictions'], val_metrics['cls_labels'],
                    plot_path, epoch, val_metrics, weight_history=combined_weight_history
                )
            
            current_r2 = val_metrics['r2_beta3']
            if current_r2 > best_r2_score:
                best_total_loss = val_total_loss
                best_r2_score = current_r2 
                logging.info(f"新的最佳模型! R²得分: {best_r2_score:.6f}, 验证总损失: {best_total_loss:.6f}")
                
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
                    
                    combined_weight_history = np.vstack(all_weight_history) if all_weight_history else None
                    save_best_multitask_outputs_gradnorm(
                        val_metrics['reg_predictions'], val_metrics['reg_labels'],
                        val_metrics['cls_predictions'], val_metrics['cls_labels'],
                        output_dir, val_metrics, weight_history=combined_weight_history
                    )
            
            logging.info(f"Epoch {epoch} 总结:")
            logging.info(f"  训练 - 总损失: {train_metrics['total_loss']:.4f}, 回归: {train_metrics['regression_loss']:.4f}, "
                        f"分类: {train_metrics['classification_loss']:.4f}, GradNorm: {train_metrics['gradnorm_loss']:.4f}")
            logging.info(f"  训练准确率: {train_metrics['train_accuracy']:.2f}%")
            logging.info(f"  验证 - 总损失: {val_total_loss:.4f}, 回归MSE: {val_metrics['regression_mse']:.6f}, "
                        f"分类准确率: {val_metrics['classification_accuracy']:.2f}%, R²: {val_metrics['r2_beta3']:.6f}")
            current_weights = val_metrics['task_weights']
            logging.info(f"  当前任务权重 - 回归: {current_weights[0]:.4f}, 分类: {current_weights[1]:.4f}")
        
        if writer:
            writer.close()
        logging.info(f"多任务GradNorm训练完成! 最佳R²得分: {best_r2_score:.6f}, 最佳验证总损失: {best_total_loss:.6f}")
        
    finally:
        sys.stdout = custom_logger.terminal
        custom_logger.close()


if __name__ == '__main__':
    main()