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
from sklearn.metrics import confusion_matrix, mean_squared_error, mean_absolute_error

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
# 2. 数据集类
# ===================================================================

class InterferenceDataset(Dataset):
    """
    改进的干涉数据集类
    - 统一的数据预处理流程
    - 更清晰的标签解析逻辑
    - 可配置的预处理选项
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
            
        # 解析标签
        self.labels = self._parse_labels()
        
        print(f"'{self.split}' 数据集初始化完成:")
        print(f"  - 文件数量: {len(self.file_paths)}")
        print(f"  - 推理模式: {self.mode}")
        if self.mode == 'classifier':
            print(f"  - 类别数量: {self.num_classes}")

    def _parse_labels(self):
        """解析beta值标签"""
        beta_pattern = re.compile(r"beta2_([\d.]+)_beta3_([\d.]+)")
        beta_pairs = []
        
        for file_path in self.file_paths:
            match = beta_pattern.search(file_path.stem)
            if match:
                beta2 = float(match.group(1))
                beta3 = float(match.group(2))
                beta_pairs.append((beta2, beta3))
            else:
                print(f"警告: 文件 {file_path.name} 无法匹配beta值, 已跳过。")

        if not beta_pairs:
            raise ValueError(f"在路径 {self.data_path} 中没有找到任何有效的数据文件!")

        # 根据推理模式生成不同格式的标签
        if self.mode == 'classifier':
            return self._create_classification_labels(beta_pairs)
        elif self.mode in ['gp', 'mlp']:
            return self._create_regression_labels(beta_pairs)
        else:
            raise ValueError(f"未知的 inference_mode: {self.mode}")

    def _create_classification_labels(self, beta_pairs):
        """创建分类任务的标签"""
        unique_pairs = sorted(list(set(beta_pairs)))
        self.class_to_pair = {i: pair for i, pair in enumerate(unique_pairs)}
        self.pair_to_class = {pair: i for i, pair in self.class_to_pair.items()}
        self.num_classes = len(unique_pairs)
        
        labels = [self.pair_to_class[pair] for pair in beta_pairs]
        print(f"分类模式: 发现 {self.num_classes} 个唯一的 (beta2, beta3) 类别")
        return labels

    def _create_regression_labels(self, beta_pairs):
        """创建回归任务的标签"""
        labels = [torch.tensor([p[0], p[1]], dtype=torch.float32) for p in beta_pairs]
        print(f"回归模式: 加载 {len(labels)} 个 (beta2, beta3) 坐标")
        return labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 加载原始数据
        file_path = self.file_paths[idx]
        image_data = np.load(file_path)
        
        # 转换为PyTorch张量并增加通道维度
        image_tensor = torch.from_numpy(image_data.astype(np.float32)).unsqueeze(0)
        
        # 应用预处理流程
        image_tensor = self.preprocessor(image_tensor)
        
        # 获取标签
        label = self.labels[idx]
        
        return image_tensor, label


# ===================================================================
# 3. 模型定义
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


class ResNetMLP(nn.Module):
    """ResNet + MLP回归模型"""
    def __init__(self, config):
        super().__init__()
        mlp_config = config['mlp_config']
        model_config = config.get('model_config', {})
        
        # 特征提取器
        feature_dim = mlp_config['feature_dim']
        dropout_rate = model_config.get('dropout_rate', 0.1)
        self.feature_extractor = ResNetFeatureExtractor(feature_dim, dropout_rate)
        
        # MLP层
        self.mlp = self._build_mlp(mlp_config, feature_dim)
        
        # 初始化权重
        self._initialize_weights()
    
    def _build_mlp(self, mlp_config, input_dim):
        """构建MLP层"""
        layers = []
        current_dim = input_dim
        
        for hidden_dim in mlp_config['hidden_dims']:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(mlp_config.get('dropout', 0.1))
            ])
            current_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(current_dim, 2))  # beta2, beta3
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """初始化权重"""
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.mlp(features)
        # 确保输出为正值
        output = F.softplus(output) + 1e-6
        return output


class SimpleVariationalGPModel(gpytorch.models.ApproximateGP):
    """简单的变分高斯过程模型"""
    def __init__(self, inducing_points, feature_dim):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0)
        )
        
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        
        super().__init__(variational_strategy)
        
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=feature_dim),
            outputscale_constraint=gpytorch.constraints.Positive()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ResNetVariationalGP(nn.Module):
    """ResNet + 变分GP联合模型"""
    def __init__(self, config, device='cpu'):
        super().__init__()
        gp_config = config['gp_config']
        model_config = config.get('model_config', {})
        
        # 特征提取器
        feature_dim = gp_config['feature_dim']
        dropout_rate = model_config.get('dropout_rate', 0.1)
        self.feature_extractor = ResNetFeatureExtractor(feature_dim, dropout_rate)
        
        # GP模型
        num_inducing = gp_config.get('num_inducing', 100)
        inducing_points = torch.randn(num_inducing, feature_dim, device=device) * 0.1
        
        self.gp_model_beta2 = SimpleVariationalGPModel(inducing_points.clone(), feature_dim)
        self.gp_model_beta3 = SimpleVariationalGPModel(inducing_points.clone(), feature_dim)
        
        # 似然函数
        self.likelihood_beta2 = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
        )
        self.likelihood_beta3 = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
        )
        
        # 初始化噪声
        self.likelihood_beta2.noise = 0.01
        self.likelihood_beta3.noise = 0.01
        
        self.feature_dim = feature_dim
        self.num_inducing = num_inducing
    
    def forward(self, x):
        features = self.feature_extractor(x)
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        dist_beta2 = self.gp_model_beta2(features)
        dist_beta3 = self.gp_model_beta3(features)
        return dist_beta2, dist_beta3


def create_model(config, num_classes=None, device='cpu'):
    """模型创建工厂函数"""
    mode = config['inference_mode']
    
    if mode == 'classifier':
        model = models.resnet34(pretrained=config['model_config']['resnet_pretrained'])
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif mode == 'mlp':
        return ResNetMLP(config)
    elif mode == 'gp':
        if not GP_AVAILABLE:
            raise ImportError("gpytorch 未安装，无法使用'gp'模式。请运行 'pip install gpytorch'")
        return ResNetVariationalGP(config, device)
    else:
        raise ValueError(f"未知的 inference_mode: {mode}")


# ===================================================================
# 4. 训练与验证函数（去除了tqdm的日志输出）
# ===================================================================

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, mode):
    """训练一个epoch"""
    model.train()
    if mode == 'gp':
        model.likelihood_beta2.train()
        model.likelihood_beta3.train()

    running_loss = 0.0
    total_samples = 0
    correct_predictions = 0
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        if mode == 'classifier':
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            correct = predicted.eq(labels).sum().item()
            correct_predictions += correct
            total_samples += labels.size(0)

        elif mode == 'mlp':
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        elif mode == 'gp':
            try:
                dist_beta2, dist_beta3 = model(inputs)
                
                labels_beta2 = labels[:, 0]
                labels_beta3 = labels[:, 1]
                
                elbo_beta2 = criterion[0](dist_beta2, labels_beta2)
                elbo_beta3 = criterion[1](dist_beta3, labels_beta3)
                
                loss = -elbo_beta2 - elbo_beta3
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"警告: 在批次 {i} 检测到异常损失值: {loss.item()}")
                    continue
                
            except Exception as e:
                print(f"训练批次 {i} 出现错误: {e}")
                continue

        running_loss += loss.item() if not (torch.isnan(loss) or torch.isinf(loss)) else 0.0

    avg_loss = running_loss / len(train_loader)
    
    if mode == 'classifier':
        train_acc = 100. * correct_predictions / total_samples if total_samples > 0 else 0.0
        print(f"Epoch {epoch} - 训练完成: 平均损失={avg_loss:.4f}, 准确率={train_acc:.2f}%")
    else:
        print(f"Epoch {epoch} - 训练完成: 平均损失={avg_loss:.4f}")
    
    return avg_loss


def validate_epoch(model, val_loader, criterion, device, epoch, mode):
    """验证一个epoch"""
    model.eval()
    if mode == 'gp':
        model.likelihood_beta2.eval()
        model.likelihood_beta3.eval()

    running_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(val_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            if mode == 'classifier':
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            elif mode == 'mlp':
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            elif mode == 'gp':
                try:
                    with gpytorch.settings.fast_pred_var():
                        dist_beta2, dist_beta3 = model(inputs)
                        
                        output_dist_beta2 = model.likelihood_beta2(dist_beta2)
                        output_dist_beta3 = model.likelihood_beta3(dist_beta3)
                        
                        labels_beta2 = labels[:, 0]
                        labels_beta3 = labels[:, 1]
                        
                        elbo_beta2 = criterion[0](dist_beta2, labels_beta2)
                        elbo_beta3 = criterion[1](dist_beta3, labels_beta3)
                        loss = -elbo_beta2 - elbo_beta3
                    
                    pred_beta2 = output_dist_beta2.mean.cpu().numpy()
                    pred_beta3 = output_dist_beta3.mean.cpu().numpy()
                    
                    if np.any(np.isnan(pred_beta2)) or np.any(np.isnan(pred_beta3)):
                        print(f"警告: 批次 {i} 产生了NaN预测")
                        continue
                    
                    batch_preds = np.column_stack([pred_beta2, pred_beta3])
                    all_preds.extend(batch_preds)
                    all_labels.extend(labels.cpu().numpy())
                    
                except Exception as e:
                    print(f"验证批次 {i} 出现错误: {e}")
                    continue
                
            if not (torch.isnan(loss) or torch.isinf(loss)):
                running_loss += loss.item()
            
    # 计算评估指标
    if mode == 'classifier':
        val_acc = 100. * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
        cm = confusion_matrix(all_labels, all_preds)
        print(f"验证完成: 损失={running_loss / len(val_loader):.4f}, 准确率={val_acc:.2f}%")
        return running_loss / len(val_loader), {'accuracy': val_acc, 'confusion_matrix': cm}

    elif mode in ['gp', 'mlp']:
        if not all_preds:
            print("警告: 没有有效的预测结果")
            return float('inf'), {'mse': float('inf'), 'mae': float('inf')}
            
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        print(f"预测值范围: Beta2=[{all_preds[:, 0].min():.4f}, {all_preds[:, 0].max():.4f}], "
              f"Beta3=[{all_preds[:, 1].min():.4f}, {all_preds[:, 1].max():.4f}]")
        
        mse = mean_squared_error(all_labels, all_preds)
        mae = mean_absolute_error(all_labels, all_preds)
        
        mse_beta2 = mean_squared_error(all_labels[:, 0], all_preds[:, 0])
        mse_beta3 = mean_squared_error(all_labels[:, 1], all_preds[:, 1])
        mae_beta2 = mean_absolute_error(all_labels[:, 0], all_preds[:, 0])
        mae_beta3 = mean_absolute_error(all_labels[:, 1], all_preds[:, 1])
        
        # 计算R²
        r2_beta2 = r2_score(all_labels[:, 0], all_preds[:, 0])
        r2_beta3 = r2_score(all_labels[:, 1], all_preds[:, 1])
        
        print(f"验证完成: 损失={running_loss / len(val_loader):.4f}, MSE={mse:.6f}, MAE={mae:.6f}")
        print(f"Beta2 - MSE: {mse_beta2:.6f}, MAE: {mae_beta2:.6f}, R²: {r2_beta2:.6f}")
        print(f"Beta3 - MSE: {mse_beta3:.6f}, MAE: {mae_beta3:.6f}, R²: {r2_beta3:.6f}")
        
        return running_loss / len(val_loader), {
            'mse': mse, 'mae': mae,
            'mse_beta2': mse_beta2, 'mse_beta3': mse_beta3,
            'mae_beta2': mae_beta2, 'mae_beta3': mae_beta3,
            'r2_beta2': r2_beta2, 'r2_beta3': r2_beta3,
            'predictions': all_preds, 'labels': all_labels
        }


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


def create_optimizer_and_criterion(model, config, train_dataset_size):
    """创建优化器和损失函数"""
    mode = config['inference_mode']
    
    # 创建损失函数
    if mode == 'classifier':
        criterion = nn.CrossEntropyLoss()
    elif mode == 'mlp':
        loss_type = config['mlp_config'].get('loss_type', 'mse')
        if loss_type == 'mse':
            criterion = nn.MSELoss()
        elif loss_type == 'mae':
            criterion = nn.L1Loss()
        elif loss_type == 'huber':
            criterion = nn.SmoothL1Loss()
        else:
            criterion = nn.MSELoss()
    elif mode == 'gp':
        criterion = [
            gpytorch.mlls.VariationalELBO(model.likelihood_beta2, model.gp_model_beta2, num_data=train_dataset_size),
            gpytorch.mlls.VariationalELBO(model.likelihood_beta3, model.gp_model_beta3, num_data=train_dataset_size)
        ]
    
    # 创建优化器
    if config.get('use_discriminative_lr', False):
        param_groups = setup_discriminative_lr(model, config)
        optimizer = optim.Adam(param_groups, weight_decay=config['weight_decay'])
    else:
        lr = config['learning_rates']['base']
        if mode == 'gp':
            lr *= 0.1  # GP使用更小的学习率
        elif mode == 'mlp':
            lr *= 0.5  # MLP使用中等学习率
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config['weight_decay'])
    
    return optimizer, criterion


def setup_discriminative_lr(model, config):
    """设置分层学习率"""
    lr_conf = config['learning_rates']
    base_lr = lr_conf['base']
    decay = lr_conf.get('layer_decay', 0.9)
    
    if isinstance(model, ResNetVariationalGP):
        feature_extractor = model.feature_extractor.features
        layer_groups = [
            list(feature_extractor[0].parameters()) + list(feature_extractor[1].parameters()) + 
            list(feature_extractor[4].parameters()) + list(feature_extractor[5].parameters()),
            list(feature_extractor[6].parameters()),
            list(feature_extractor[7].parameters()),
        ]
        
        gp_params = list(model.feature_extractor.feature_proj.parameters()) + \
                   list(model.gp_model_beta2.parameters()) + \
                   list(model.gp_model_beta3.parameters()) + \
                   list(model.likelihood_beta2.parameters()) + \
                   list(model.likelihood_beta3.parameters())
        
        return [
            {'params': layer_groups[0], 'lr': base_lr * (decay ** 3)},
            {'params': layer_groups[1], 'lr': base_lr * (decay ** 2)},
            {'params': layer_groups[2], 'lr': base_lr * decay},
            {'params': gp_params, 'lr': base_lr}
        ]
    else:  # Handles ResNetMLP and standard ResNet for classification
        # 正确地访问 ResNetMLP 内部的 ResNet 骨干网络
        resnet_backbone = model.feature_extractor.features
        
        # 模型的 "头部" (head) 包含特征映射层和最终的 MLP 回归层
        head_layers = list(model.feature_extractor.feature_proj.parameters()) + \
                      list(model.mlp.parameters())
        
        # 将 ResNet 骨干网络分层 (此分组适用于 ResNet18/34)
        layer_groups = [
            # 早期层
            list(resnet_backbone[0].parameters()) + list(resnet_backbone[1].parameters()) + 
            list(resnet_backbone[4].parameters()) + list(resnet_backbone[5].parameters()),
            # 中期层
            list(resnet_backbone[6].parameters()),
            # 后期层
            list(resnet_backbone[7].parameters()),
        ]
        
        return [
            {'params': layer_groups[0], 'lr': base_lr * (decay ** 3)},
            {'params': layer_groups[1], 'lr': base_lr * (decay ** 2)},
            {'params': layer_groups[2], 'lr': base_lr * decay},
            {'params': head_layers, 'lr': base_lr} # 头部使用基础学习率
        ]


def plot_confusion_matrix(cm, class_names, output_path):
    """绘制混淆矩阵"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_confusion_matrix_to_tensorboard(writer, cm, class_names, epoch):
    """保存混淆矩阵到TensorBoard"""
    fig = plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    writer.add_figure('Validation/Confusion_Matrix', fig, epoch)
    plt.close()


def plot_regression_predictions(predictions, labels, output_path, epoch, mode_name, mse_beta2, mse_beta3, r2_beta2, r2_beta3):
    """绘制回归预测结果的散点图，带有MSE和R²标注"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Beta2 预测 vs 真实值
    axes[0].scatter(labels[:, 0], predictions[:, 0], alpha=0.6)
    axes[0].plot([labels[:, 0].min(), labels[:, 0].max()], 
                 [labels[:, 0].min(), labels[:, 0].max()], 'r--', lw=2)
    axes[0].set_xlabel('True Beta2')
    axes[0].set_ylabel('Predicted Beta2')
    axes[0].set_title(f'{mode_name} Beta2 Predictions vs True Values')
    axes[0].grid(True, alpha=0.3)
    
    # 在左上角添加MSE和R²标注
    axes[0].text(0.05, 0.95, f'MSE: {mse_beta2:.6f}\nR²: {r2_beta2:.6f}', 
                transform=axes[0].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Beta3 预测 vs 真实值
    axes[1].scatter(labels[:, 1], predictions[:, 1], alpha=0.6)
    axes[1].plot([labels[:, 1].min(), labels[:, 1].max()], 
                 [labels[:, 1].min(), labels[:, 1].max()], 'r--', lw=2)
    axes[1].set_xlabel('True Beta3')
    axes[1].set_ylabel('Predicted Beta3')
    axes[1].set_title(f'{mode_name} Beta3 Predictions vs True Values')
    axes[1].grid(True, alpha=0.3)
    
    # 在左上角添加MSE和R²标注
    axes[1].text(0.05, 0.95, f'MSE: {mse_beta3:.6f}\nR²: {r2_beta3:.6f}', 
                transform=axes[1].transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2D散点图：(beta2, beta3)空间中的预测
    axes[2].scatter(labels[:, 0], labels[:, 1], alpha=0.6, label='True', s=30)
    axes[2].scatter(predictions[:, 0], predictions[:, 1], alpha=0.6, label='Predicted', s=30)
    axes[2].set_xlabel('Beta2')
    axes[2].set_ylabel('Beta3')
    axes[2].set_title(f'{mode_name} Predictions in (Beta2, Beta3) Space')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_best_model_outputs(predictions, labels, output_dir, mode_name, val_metrics):
    """保存最佳模型的输出数据和图片"""
    # 保存图片
    plot_path = output_dir / 'best_model_output.png'
    plot_regression_predictions(
        predictions, 
        labels, 
        plot_path, 
        epoch=None,  # 不需要epoch信息 
        mode_name=mode_name.upper(),
        mse_beta2=val_metrics['mse_beta2'],
        mse_beta3=val_metrics['mse_beta3'],
        r2_beta2=val_metrics['r2_beta2'],
        r2_beta3=val_metrics['r2_beta3']
    )
    
    # 保存数据到npz文件
    npz_path = output_dir / 'best_model_output.npz'
    np.savez(
        npz_path,
        predictions=predictions,
        labels=labels,
        mse_beta2=val_metrics['mse_beta2'],
        mse_beta3=val_metrics['mse_beta3'],
        mae_beta2=val_metrics['mae_beta2'],
        mae_beta3=val_metrics['mae_beta3'],
        r2_beta2=val_metrics['r2_beta2'],
        r2_beta3=val_metrics['r2_beta3'],
        overall_mse=val_metrics['mse'],
        overall_mae=val_metrics['mae']
    )
    
    print(f"最佳模型输出已保存:")
    print(f"  - 图片: {plot_path}")
    print(f"  - 数据: {npz_path}")


# ===================================================================
# 6. 主函数
# ===================================================================

def main():
    """主训练函数"""
    # 加载配置
    with open('config2.yaml', 'r', encoding='utf-8') as f:
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
                logging.StreamHandler()  # 这会输出到我们重定向的stdout
            ]
        )
        
        # TensorBoard写入器
        if config['logging']['tensorboard']:
            writer = SummaryWriter(str(logs_dir))
        else:
            writer = None
        
        # 创建数据集
        train_dataset = InterferenceDataset(
            Path(config['data_path']) / 'train', 
            config=config, 
            split='train'
        )
        val_dataset = InterferenceDataset(
            Path(config['data_path']) / 'val', 
            config=config, 
            split='val'
        )
        
        # 创建模型
        mode = config['inference_mode']
        if mode == 'classifier':
            model = create_model(config, num_classes=train_dataset.num_classes, device=device)
            class_names = [f"({p[0]:.4f}, {p[1]:.4f})" for p in train_dataset.class_to_pair.values()]
            best_metric = 0.0  # Accuracy
            metric_mode = 'max'
        else:
            model = create_model(config, device=device)
            best_metric = float('inf')  # MSE
            metric_mode = 'min'
        
        model.to(device)
        
        # 创建优化器和损失函数
        optimizer, criterion = create_optimizer_and_criterion(model, config, len(train_dataset))
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=metric_mode, factor=0.5, patience=5, verbose=True
        )
        
        # 最佳模型路径
        best_model_path = checkpoints_dir / 'best_model.pth'
        
        # 开始训练
        logging.info(f"开始训练... 模式: {mode.upper()}")
        if mode == 'gp':
            logging.info(f"变分GP配置: 特征维度={model.feature_dim}, 诱导点数量={model.num_inducing}")
        elif mode == 'mlp':
            mlp_conf = config['mlp_config']
            logging.info(f"MLP回归配置: 特征维度={mlp_conf['feature_dim']}, 隐藏层={mlp_conf['hidden_dims']}")
        
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
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, mode)
            val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device, epoch, mode)
            
            # 学习率调度
            if mode == 'classifier':
                scheduler_metric = val_metrics['accuracy']
            else:
                scheduler_metric = val_metrics['mse']
            scheduler.step(scheduler_metric)
            
            # 记录到TensorBoard
            if writer:
                writer.add_scalar('Loss/Train', train_loss, epoch)
                writer.add_scalar('Loss/Validation', val_loss, epoch)
                writer.add_scalar('Learning_Rate', optimizer.param_groups[-1]['lr'], epoch)
                
                if mode == 'classifier':
                    writer.add_scalar('Accuracy/Validation', val_metrics['accuracy'], epoch)
                    save_confusion_matrix_to_tensorboard(writer, val_metrics['confusion_matrix'], class_names, epoch)
                else:
                    writer.add_scalar('MSE/Validation', val_metrics['mse'], epoch)
                    writer.add_scalar('MAE/Validation', val_metrics['mae'], epoch)
                    writer.add_scalar('MSE_Beta2/Validation', val_metrics['mse_beta2'], epoch)
                    writer.add_scalar('MSE_Beta3/Validation', val_metrics['mse_beta3'], epoch)
                    writer.add_scalar('R2_Beta2/Validation', val_metrics['r2_beta2'], epoch)
                    writer.add_scalar('R2_Beta3/Validation', val_metrics['r2_beta3'], epoch)
                    
                    if mode == 'gp':
                        writer.add_scalar('GP/Noise_Beta2', model.likelihood_beta2.noise.item(), epoch)
                        writer.add_scalar('GP/Noise_Beta3', model.likelihood_beta3.noise.item(), epoch)
            
            # 绘制预测结果（定期）
            if mode in ['gp', 'mlp'] and config['validation']['plot_predictions']:
                if epoch % config['validation']['plot_interval'] == 0:
                    plot_path = plots_dir / f'{mode.lower()}_predictions_epoch_{epoch}.png'
                    plot_regression_predictions(
                        val_metrics['predictions'], 
                        val_metrics['labels'], 
                        plot_path, 
                        epoch, 
                        mode.upper(),
                        val_metrics['mse_beta2'],
                        val_metrics['mse_beta3'],
                        val_metrics['r2_beta2'],
                        val_metrics['r2_beta3']
                    )
            
            # 保存最佳模型
            is_best = False
            if mode == 'classifier':
                if val_metrics['accuracy'] > best_metric:
                    best_metric = val_metrics['accuracy']
                    is_best = True
                    logging.info(f"新的最佳模型! 验证准确率: {best_metric:.2f}%")
            else:
                if val_metrics['mse'] < best_metric:
                    best_metric = val_metrics['mse']
                    is_best = True
                    logging.info(f"新的最佳模型! 验证MSE: {best_metric:.6f}")
            
            if is_best and config['save_options']['save_best_only']:
                save_payload = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_metric': best_metric,
                    'config': config
                }
                if mode == 'classifier':
                    save_payload['class_to_pair'] = train_dataset.class_to_pair
                torch.save(save_payload, best_model_path)
                
                # 保存最佳模型的输出（对于回归任务）
                if mode in ['gp', 'mlp']:
                    save_best_model_outputs(
                        val_metrics['predictions'], 
                        val_metrics['labels'], 
                        output_dir, 
                        mode, 
                        val_metrics
                    )
            
            # 输出训练日志
            if mode == 'classifier':
                logging.info(f"Epoch {epoch} -> Train Loss: {train_loss:.4f} | "
                            f"Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
            else:
                logging.info(f"Epoch {epoch} -> Train Loss: {train_loss:.4f} | "
                            f"Val Loss: {val_loss:.4f}, MSE: {val_metrics['mse']:.6f}, "
                            f"MAE: {val_metrics['mae']:.6f}")
        
        if writer:
            writer.close()
        logging.info(f"训练完成! 最佳验证指标: {best_metric}")
        
    finally:
        # 确保恢复标准输出并关闭日志文件
        sys.stdout = custom_logger.terminal
        custom_logger.close()


if __name__ == '__main__':
    main()