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

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
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
# 1. 数据集类
# ===================================================================
class InterferenceDataset(Dataset):
    """
    自定义数据集，用于加载耦合beta值的.npy文件
    - 能够从文件名中同时解析 beta2 和 beta3
    - 根据 config 中的 'inference_mode' 返回不同格式的标签
    """
    def __init__(self, data_path, config, transform=None, split='train'):
        self.data_path = Path(data_path)
        self.config = config
        self.transform = transform
        self.split = split
        self.mode = self.config['inference_mode']

        self.file_paths = sorted(list(self.data_path.glob('**/*.npy')))
        self.labels = []
        
        # 新的正则表达式，用于匹配 beta2 和 beta3
        beta_pattern = re.compile(r"beta2_([\d.]+)_beta3_([\d.]+)")

        # 扫描文件并提取beta值对
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

        if self.mode == 'classifier':
            # 分类模式: 将每个 (beta2, beta3) 对映射到一个唯一的整数类别
            unique_pairs = sorted(list(set(beta_pairs)))
            self.class_to_pair = {i: pair for i, pair in enumerate(unique_pairs)}
            self.pair_to_class = {pair: i for i, pair in self.class_to_pair.items()}
            self.num_classes = len(unique_pairs)
            self.labels = [self.pair_to_class[pair] for pair in beta_pairs]
            print(f"'{self.split}' 集 (Classifier Mode): 动态发现 {self.num_classes} 个唯一的 (beta2, beta3) 类别。")

        elif self.mode == 'gp':
            # GP模式: 标签就是 (beta2, beta3) 的浮点数对
            raw_labels = [torch.tensor([p[0], p[1]], dtype=torch.float32) for p in beta_pairs]
            
            # 检查数据范围并打印统计信息
            all_beta2 = [p[0] for p in beta_pairs]
            all_beta3 = [p[1] for p in beta_pairs]
            print(f"'{self.split}' 集 Beta2 范围: [{min(all_beta2):.4f}, {max(all_beta2):.4f}]")
            print(f"'{self.split}' 集 Beta3 范围: [{min(all_beta3):.4f}, {max(all_beta3):.4f}]")
            
            # 数据标准化：确保beta值在合理范围内
            # 如果beta值过大或过小，可能导致数值不稳定
            if max(all_beta2) > 10 or max(all_beta3) > 10 or min(all_beta2) < -10 or min(all_beta3) < -10:
                print(f"警告: Beta值范围过大，建议进行标准化")
                
            self.labels = raw_labels
            self.num_classes = None # GP模式没有类别概念
            print(f"'{self.split}' 集 (GP Mode): 加载了 {len(self.labels)} 个 (beta2, beta3) 坐标。")
        else:
            raise ValueError(f"未知的 inference_mode: {self.mode}")

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image_np = np.load(file_path)
        label = self.labels[idx]
        
        image = torch.from_numpy(image_np).float().unsqueeze(0)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# ===================================================================
# 2. 模型定义
# ===================================================================

# ResNet特征提取器 - 增加正则化
class ResNetFeatureExtractor(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        resnet = models.resnet34(pretrained=True)
        # 移除原始的fc层
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # 添加dropout和batch normalization来稳定训练
        self.feature_proj = nn.Sequential(
            nn.Linear(resnet.fc.in_features, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.feature_proj(x)
        return x

# 改进的变分高斯过程模型
class SimpleVariationalGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, feature_dim):
        # 使用更稳定的变分分布
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0)
        )
        
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        
        super().__init__(variational_strategy)
        
        # 使用常数均值和RBF核
        self.mean_module = gpytorch.means.ConstantMean()
        
        # 改进核函数配置，增加数值稳定性
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=feature_dim),
            outputscale_constraint=gpytorch.constraints.Positive()
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# ResNet + 变分GP联合模型 - 改进版本
class ResNetVariationalGP(nn.Module):
    def __init__(self, feature_dim, output_dim, num_inducing=100, device='cpu'):
        super().__init__()
        self.feature_extractor = ResNetFeatureExtractor(feature_dim)
        
        # 更好的诱导点初始化
        inducing_points = torch.randn(num_inducing, feature_dim, device=device) * 0.1
        
        # 创建两个独立的GP模型，分别预测beta2和beta3
        self.gp_model_beta2 = SimpleVariationalGPModel(inducing_points.clone(), feature_dim)
        self.gp_model_beta3 = SimpleVariationalGPModel(inducing_points.clone(), feature_dim)
        
        # 创建高斯似然函数，设置合理的噪声先验
        self.likelihood_beta2 = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
        )
        self.likelihood_beta3 = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.GreaterThan(1e-6)
        )
        
        # 初始化似然噪声
        self.likelihood_beta2.noise = 0.01
        self.likelihood_beta3.noise = 0.01
        
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.num_inducing = num_inducing
    
    def forward(self, x):
        features = self.feature_extractor(x)
        # 对特征进行L2归一化，提高稳定性
        features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        # 返回两个独立的分布
        dist_beta2 = self.gp_model_beta2(features)
        dist_beta3 = self.gp_model_beta3(features)
        return dist_beta2, dist_beta3

# 模型创建工厂函数
def create_model(config, num_classes=None, train_dataset=None, device='cpu'):
    mode = config['inference_mode']
    if mode == 'classifier':
        model = models.resnet34(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif mode == 'gp':
        if not GP_AVAILABLE:
            raise ImportError("gpytorch 未安装，无法使用'gp'模式。请运行 'pip install gpytorch'")
        
        gp_conf = config['gp_config']
        model = ResNetVariationalGP(
            feature_dim=gp_conf['feature_dim'], 
            output_dim=gp_conf['output_dim'],
            num_inducing=gp_conf.get('num_inducing', 100),
            device=device
        )
        
        print(f"创建变分GP模型: 特征维度={gp_conf['feature_dim']}, "
              f"输出维度={gp_conf['output_dim']}, 诱导点数量={gp_conf.get('num_inducing', 100)}")
        return model
    else:
        raise ValueError(f"未知的 inference_mode: {mode}")

# ===================================================================
# 3. 训练与验证循环
# ===================================================================

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, mode):
    model.train()
    if mode == 'gp':
        model.likelihood_beta2.train()
        model.likelihood_beta3.train()

    running_loss = 0.0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch} - Training')
    
    for i, (inputs, labels) in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        if mode == 'classifier':
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = outputs.max(1)
            correct = predicted.eq(labels).sum().item()
            total = labels.size(0)
            pbar.set_postfix({'Loss': f'{loss.item():.4f}', 'Acc': f'{100.*correct/total:.2f}%'})

        elif mode == 'gp':
            try:
                # 变分GP的前向传播
                dist_beta2, dist_beta3 = model(inputs)
                
                # 分别计算每个输出的ELBO损失
                labels_beta2 = labels[:, 0]
                labels_beta3 = labels[:, 1]
                
                # 计算ELBO损失 (注意：ELBO本身是对数似然，我们要最大化它)
                # 所以损失应该是 -ELBO
                elbo_beta2 = criterion[0](dist_beta2, labels_beta2)
                elbo_beta3 = criterion[1](dist_beta3, labels_beta3)
                
                # 总损失是负ELBO的和
                loss = -elbo_beta2 - elbo_beta3
                
                # 梯度裁剪防止梯度爆炸
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                # 检查损失是否为NaN或无穷大
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"警告: 在批次 {i} 检测到异常损失值: {loss.item()}")
                    continue
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}', 
                    'ELBO_β2': f'{elbo_beta2.item():.4f}', 
                    'ELBO_β3': f'{elbo_beta3.item():.4f}',
                    'Noise_β2': f'{model.likelihood_beta2.noise.item():.4f}',
                    'Noise_β3': f'{model.likelihood_beta3.noise.item():.4f}'
                })
                
            except Exception as e:
                print(f"训练批次 {i} 出现错误: {e}")
                continue

        running_loss += loss.item() if not (torch.isnan(loss) or torch.isinf(loss)) else 0.0

    return running_loss / len(train_loader)

def validate_epoch(model, val_loader, criterion, device, epoch, mode):
    model.eval()
    if mode == 'gp':
        model.likelihood_beta2.eval()
        model.likelihood_beta3.eval()

    running_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f'Epoch {epoch} - Validation')
        for i, (inputs, labels) in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            if mode == 'classifier':
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
            elif mode == 'gp':
                try:
                    # 在评估模式下，使用快速预测
                    with gpytorch.settings.fast_pred_var():
                        # 变分GP的预测
                        dist_beta2, dist_beta3 = model(inputs)
                        
                        # 获取似然预测
                        output_dist_beta2 = model.likelihood_beta2(dist_beta2)
                        output_dist_beta3 = model.likelihood_beta3(dist_beta3)
                        
                        # 计算ELBO损失
                        labels_beta2 = labels[:, 0]
                        labels_beta3 = labels[:, 1]
                        
                        elbo_beta2 = criterion[0](dist_beta2, labels_beta2)
                        elbo_beta3 = criterion[1](dist_beta3, labels_beta3)
                        loss = -elbo_beta2 - elbo_beta3
                    
                    # 预测的均值
                    pred_beta2 = output_dist_beta2.mean.cpu().numpy()
                    pred_beta3 = output_dist_beta3.mean.cpu().numpy()
                    
                    # 检查预测值是否合理
                    if np.any(np.isnan(pred_beta2)) or np.any(np.isnan(pred_beta3)):
                        print(f"警告: 批次 {i} 产生了NaN预测")
                        continue
                    
                    # 组合预测结果
                    batch_preds = np.column_stack([pred_beta2, pred_beta3])
                    all_preds.extend(batch_preds)
                    all_labels.extend(labels.cpu().numpy())
                    
                except Exception as e:
                    print(f"验证批次 {i} 出现错误: {e}")
                    continue
                
            if not (torch.isnan(loss) or torch.isinf(loss)):
                running_loss += loss.item()
            
    # 计算并返回评估结果
    if mode == 'classifier':
        val_acc = 100. * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
        cm = confusion_matrix(all_labels, all_preds)
        print(f"Validation Acc: {val_acc:.2f}%")
        return running_loss / len(val_loader), {'accuracy': val_acc, 'confusion_matrix': cm}

    elif mode == 'gp':
        if not all_preds:
            print("警告: 没有有效的预测结果")
            return float('inf'), {'mse': float('inf'), 'mae': float('inf')}
            
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # 数据质量检查
        print(f"预测值范围: Beta2=[{all_preds[:, 0].min():.4f}, {all_preds[:, 0].max():.4f}], "
              f"Beta3=[{all_preds[:, 1].min():.4f}, {all_preds[:, 1].max():.4f}]")
        
        mse = mean_squared_error(all_labels, all_preds)
        mae = mean_absolute_error(all_labels, all_preds)
        
        # 分别计算beta2和beta3的误差
        mse_beta2 = mean_squared_error(all_labels[:, 0], all_preds[:, 0])
        mse_beta3 = mean_squared_error(all_labels[:, 1], all_preds[:, 1])
        mae_beta2 = mean_absolute_error(all_labels[:, 0], all_preds[:, 0])
        mae_beta3 = mean_absolute_error(all_labels[:, 1], all_preds[:, 1])
        
        print(f"Validation MSE: {mse:.6f}, MAE: {mae:.6f}")
        print(f"Beta2 - MSE: {mse_beta2:.6f}, MAE: {mae_beta2:.6f}")
        print(f"Beta3 - MSE: {mse_beta3:.6f}, MAE: {mae_beta3:.6f}")
        
        return running_loss / len(val_loader), {
            'mse': mse, 'mae': mae,
            'mse_beta2': mse_beta2, 'mse_beta3': mse_beta3,
            'mae_beta2': mae_beta2, 'mae_beta3': mae_beta3,
            'predictions': all_preds, 'labels': all_labels
        }

# ===================================================================
# 4. 辅助函数
# ===================================================================

class PreprocessTransform:
    def __init__(self, resize_dim=224):
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.resize = transforms.Resize((resize_dim, resize_dim))
    
    def __call__(self, image):
        image = self.resize(image)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)
        image = self.normalize(image)
        return image

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

def plot_confusion_matrix(cm, class_names, output_path):
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_confusion_matrix_to_tensorboard(writer, cm, class_names, epoch):
    fig = plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    writer.add_figure('Validation/Confusion_Matrix', fig, epoch)
    plt.close()

def plot_gp_predictions(predictions, labels, output_path, epoch):
    """绘制GP预测结果的散点图"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Beta2 预测 vs 真实值
    axes[0].scatter(labels[:, 0], predictions[:, 0], alpha=0.6)
    axes[0].plot([labels[:, 0].min(), labels[:, 0].max()], 
                 [labels[:, 0].min(), labels[:, 0].max()], 'r--', lw=2)
    axes[0].set_xlabel('True Beta2')
    axes[0].set_ylabel('Predicted Beta2')
    axes[0].set_title('Beta2 Predictions vs True Values')
    axes[0].grid(True, alpha=0.3)
    
    # Beta3 预测 vs 真实值
    axes[1].scatter(labels[:, 1], predictions[:, 1], alpha=0.6)
    axes[1].plot([labels[:, 1].min(), labels[:, 1].max()], 
                 [labels[:, 1].min(), labels[:, 1].max()], 'r--', lw=2)
    axes[1].set_xlabel('True Beta3')
    axes[1].set_ylabel('Predicted Beta3')
    axes[1].set_title('Beta3 Predictions vs True Values')
    axes[1].grid(True, alpha=0.3)
    
    # 2D散点图：(beta2, beta3)空间中的预测
    axes[2].scatter(labels[:, 0], labels[:, 1], alpha=0.6, label='True', s=30)
    axes[2].scatter(predictions[:, 0], predictions[:, 1], alpha=0.6, label='Predicted', s=30)
    axes[2].set_xlabel('Beta2')
    axes[2].set_ylabel('Beta3')
    axes[2].set_title('Predictions in (Beta2, Beta3) Space')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path / f'gp_predictions_epoch_{epoch}.png', dpi=150)
    plt.close()

def setup_discriminative_lr(model, config):
    lr_conf = config['learning_rates']
    base_lr = lr_conf['base']
    decay = lr_conf.get('layer_decay', 0.9)
    
    # 根据模型类型分配参数
    if isinstance(model, ResNetVariationalGP):
        # ResNet特征提取器的层
        feature_extractor = model.feature_extractor.features
        layer_groups = [
            list(feature_extractor[0].parameters()) + list(feature_extractor[1].parameters()) + 
            list(feature_extractor[4].parameters()) + list(feature_extractor[5].parameters()),  # conv1, bn1, layer1, layer2
            list(feature_extractor[6].parameters()),  # layer3
            list(feature_extractor[7].parameters()),  # layer4
        ]
        
        # GP和投影层的参数 - 使用较小的学习率
        gp_params = list(model.feature_extractor.feature_proj.parameters()) + \
                   list(model.gp_model_beta2.parameters()) + \
                   list(model.gp_model_beta3.parameters()) + \
                   list(model.likelihood_beta2.parameters()) + \
                   list(model.likelihood_beta3.parameters())
        
        return [
            {'params': layer_groups[0], 'lr': base_lr * (decay ** 3)},
            {'params': layer_groups[1], 'lr': base_lr * (decay ** 2)},
            {'params': layer_groups[2], 'lr': base_lr * decay},
            {'params': gp_params, 'lr': base_lr * 0.1}  # GP参数使用更小的学习率
        ]
    else:  # ResNet for classification
        feature_extractor = nn.Sequential(*list(model.children())[:-1])
        classifier_layers = list(model.fc.parameters())
        
        layer_groups = [
            list(feature_extractor[0].parameters()) + list(feature_extractor[1].parameters()) + 
            list(feature_extractor[4].parameters()) + list(feature_extractor[5].parameters()),
            list(feature_extractor[6].parameters()),
            list(feature_extractor[7].parameters()),
        ]
        
        return [
            {'params': layer_groups[0], 'lr': base_lr * (decay ** 3)},
            {'params': layer_groups[1], 'lr': base_lr * (decay ** 2)},
            {'params': layer_groups[2], 'lr': base_lr * decay},
            {'params': classifier_layers, 'lr': base_lr}
        ]

# ===================================================================
# 5. 主函数
# ===================================================================

def main():
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    set_seed(config['seed'])
    device = get_device(config['device'])
    print(f"使用设备: {device}")
    
    output_dir = Path(config['output_dir']) / config['experiment_name']
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = output_dir / 'logs'; logs_dir.mkdir(exist_ok=True)
    checkpoints_dir = output_dir / 'checkpoints'; checkpoints_dir.mkdir(exist_ok=True)
    plots_dir = output_dir / 'plots'; plots_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                        handlers=[logging.FileHandler(output_dir / 'training.log'), logging.StreamHandler()])
    
    writer = SummaryWriter(str(logs_dir))
    
    transform = PreprocessTransform()
    
    train_dataset = InterferenceDataset(Path(config['data_path']) / 'train', config=config, transform=transform, split='train')
    val_dataset = InterferenceDataset(Path(config['data_path']) / 'val', config=config, transform=transform, split='val')
    
    mode = config['inference_mode']
    
    if mode == 'classifier':
        model = create_model(config, num_classes=train_dataset.num_classes)
        criterion = nn.CrossEntropyLoss()
        class_names = [f"({p[0]:.4f}, {p[1]:.4f})" for p in train_dataset.class_to_pair.values()]
        best_metric = 0.0  # Accuracy
        metric_mode = 'max'
    elif mode == 'gp':
        model = create_model(config, train_dataset=train_dataset, device=device)
        # 为两个独立的GP创建两个独立的ELBO损失函数
        criterion = [
            gpytorch.mlls.VariationalELBO(model.likelihood_beta2, model.gp_model_beta2, num_data=len(train_dataset)),
            gpytorch.mlls.VariationalELBO(model.likelihood_beta3, model.gp_model_beta3, num_data=len(train_dataset))
        ]
        best_metric = float('inf')  # MSE
        metric_mode = 'min'
    else:
        raise ValueError("Invalid mode")

    model.to(device)
    
    # 设置优化器 - 使用更保守的学习率
    if config.get('use_discriminative_lr', False):
        param_groups = setup_discriminative_lr(model, config)
        optimizer = optim.Adam(param_groups, weight_decay=config['weight_decay'])
    else:
        # 为GP模式使用更小的学习率
        if mode == 'gp':
            lr = config['learning_rates']['base'] * 0.1  # GP使用更小的学习率
        else:
            lr = config['learning_rates']['base']
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=config['weight_decay'])
        
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=metric_mode, factor=0.5, 
                                                    patience=5, verbose=True)
    
    best_model_path = checkpoints_dir / 'best_model.pth'
    
    logging.info(f"开始训练... 模式: {mode.upper()}")
    if mode == 'gp':
        logging.info(f"变分GP配置: 特征维度={model.feature_dim}, 输出维度={model.output_dim}, 诱导点数量={model.num_inducing}")

    for epoch in range(1, config['epochs'] + 1):
        logging.info(f"\n{'='*50}\nEpoch {epoch}/{config['epochs']}\n{'='*50}")
        
        # 训练
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, 
                                 num_workers=config['num_workers'])
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, 
                               num_workers=config['num_workers'])
        
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, mode)
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device, epoch, mode)
        
        # 学习率调度
        if mode == 'classifier':
            scheduler_metric = val_metrics['accuracy']
        else:  # GP mode
            scheduler_metric = val_metrics['mse']
        scheduler.step(scheduler_metric)
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[-1]['lr'], epoch)
        
        is_best = False
        if mode == 'classifier':
            val_acc = val_metrics['accuracy']
            writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            save_confusion_matrix_to_tensorboard(writer, val_metrics['confusion_matrix'], class_names, epoch)
            if val_acc > best_metric:
                best_metric = val_acc
                is_best = True
                logging.info(f"新的最佳模型! 验证准确率: {best_metric:.2f}%")
        
        elif mode == 'gp':
            val_mse = val_metrics['mse']
            writer.add_scalar('MSE/Validation', val_mse, epoch)
            writer.add_scalar('MAE/Validation', val_metrics['mae'], epoch)
            writer.add_scalar('MSE_Beta2/Validation', val_metrics['mse_beta2'], epoch)
            writer.add_scalar('MSE_Beta3/Validation', val_metrics['mse_beta3'], epoch)
            writer.add_scalar('MAE_Beta2/Validation', val_metrics['mae_beta2'], epoch)
            writer.add_scalar('MAE_Beta3/Validation', val_metrics['mae_beta3'], epoch)
            
            # 记录噪声参数到TensorBoard
            writer.add_scalar('GP/Noise_Beta2', model.likelihood_beta2.noise.item(), epoch)
            writer.add_scalar('GP/Noise_Beta3', model.likelihood_beta3.noise.item(), epoch)
            
            # 每5个epoch保存一次预测图
            if epoch % 5 == 0:
                plot_gp_predictions(val_metrics['predictions'], val_metrics['labels'], plots_dir, epoch)
            
            if val_mse < best_metric:
                best_metric = val_mse
                is_best = True
                logging.info(f"新的最佳模型! 验证MSE: {best_metric:.6f}")

        # 保存最佳模型
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
        
        # 日志输出
        if mode == 'classifier':
            logging.info(f"Epoch {epoch} Results -> Train Loss: {train_loss:.4f} | "
                        f"Val Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
        else:
            logging.info(f"Epoch {epoch} Results -> Train Loss: {train_loss:.4f} | "
                        f"Val Loss: {val_loss:.4f}, MSE: {val_metrics['mse']:.6f}, MAE: {val_metrics['mae']:.6f}")

    writer.close()
    logging.info(f"训练完成! 最佳验证指标: {best_metric}")

if __name__ == '__main__':
    main()