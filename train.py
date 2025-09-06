import os
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import logging


class InterferenceDataset(Dataset):
    """
    自定义数据集类，用于加载.npy格式的干涉条纹图像
    """
    def __init__(self, data_path, transform=None, split='train'):
        self.data_path = Path(data_path)
        self.transform = transform
        self.split = split
        
        # 获取所有.npy文件
        self.file_paths = list(self.data_path.glob('**/*.npy'))
        
        # 从文件名中提取类别标签（假设文件名格式为: classX_xxx.npy）
        self.labels = []
        self.valid_files = []
        
        for file_path in self.file_paths:
            try:
                # 提取类别标签，假设文件名包含class0, class1, class2, class3
                filename = file_path.stem.lower()
                if 'class0' in filename:
                    label = 0
                elif 'class1' in filename:
                    label = 1
                elif 'class2' in filename:
                    label = 2
                elif 'class3' in filename:
                    label = 3
                else:
                    continue  # 跳过无法识别类别的文件
                
                self.labels.append(label)
                self.valid_files.append(file_path)
            except Exception as e:
                print(f"跳过文件 {file_path}: {e}")
                continue
        
        print(f"{split}集加载完成: {len(self.valid_files)} 个文件")
        print(f"类别分布: {np.bincount(self.labels)}")
    
    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        # 加载.npy文件
        file_path = self.valid_files[idx]
        image = np.load(file_path)
        label = self.labels[idx]
        
        # 确保图像是645x645
        if image.shape != (645, 645):
            raise ValueError(f"图像尺寸错误: {image.shape}, 期望: (645, 645)")
        
        # 转换为torch tensor并添加通道维度
        image = torch.from_numpy(image).float().unsqueeze(0)  # (1, 645, 645)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


class PreprocessTransform:
    """
    预处理变换类：将645x645单通道图像转换为224x224三通道图像
    """
    def __init__(self):
        # ImageNet标准化参数
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        self.resize = transforms.Resize((224, 224))
    
    def __call__(self, image):
        # image shape: (1, 645, 645)
        
        # 缩放到224x224
        image = self.resize(image)  # (1, 224, 224)
        
        # 复制单通道为三通道
        image = image.repeat(3, 1, 1)  # (3, 224, 224)
        
        # 标准化
        image = self.normalize(image)
        
        return image


def set_seed(seed):
    """设置随机种子以确保实验可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(config_device):
    """根据配置确定训练设备"""
    if config_device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(config_device)
    
    if device.type == 'cuda' and not torch.cuda.is_available():
        print("CUDA不可用，回退到CPU")
        device = torch.device('cpu')
    
    return device


def create_model(num_classes=4):
    """创建ResNet34模型"""
    # 使用旧版API加载预训练模型
    model = models.resnet34(pretrained=True)
    
    # 修改最后一层为4分类
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model


def setup_discriminative_lr(model, base_lr=1e-3, layer_lr_decay=0.1):
    """
    设置判别性学习率
    底层使用较小的学习率，高层使用较大的学习率
    """
    # 获取所有层的参数组
    layer_groups = []
    
    # 底层特征提取层 (conv1, bn1, layer1, layer2)
    early_layers = []
    early_layers.extend(list(model.conv1.parameters()))
    early_layers.extend(list(model.bn1.parameters()))
    early_layers.extend(list(model.layer1.parameters()))
    early_layers.extend(list(model.layer2.parameters()))
    
    # 中层 (layer3)
    mid_layers = list(model.layer3.parameters())
    
    # 高层 (layer4)
    high_layers = list(model.layer4.parameters())
    
    # 分类头 (fc)
    classifier = list(model.fc.parameters())
    
    # 设置不同的学习率
    layer_groups = [
        {'params': early_layers, 'lr': base_lr * (layer_lr_decay ** 3)},
        {'params': mid_layers, 'lr': base_lr * (layer_lr_decay ** 2)},
        {'params': high_layers, 'lr': base_lr * layer_lr_decay},
        {'params': classifier, 'lr': base_lr}
    ]
    
    return layer_groups


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} - Training')
    
    for inputs, labels in pbar:
        # 将数据移到指定设备
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 更新进度条
        pbar.set_postfix({
            'Loss': f'{running_loss/len(pbar.container):.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion, device, epoch):
    """验证一个epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predicted = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} - Validation')
        
        for inputs, labels in pbar:
            # 将数据移到指定设备
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 前向传播
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 收集预测结果用于混淆矩阵
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{running_loss/len(pbar.container):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_predicted)
    
    return epoch_loss, epoch_acc, cm


def plot_confusion_matrix(cm, class_names, output_path):
    """绘制混淆矩阵"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_confusion_matrix_to_tensorboard(writer, cm, epoch):
    """将混淆矩阵保存到TensorBoard"""
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'],
                yticklabels=['Class 0', 'Class 1', 'Class 2', 'Class 3'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    writer.add_figure('Validation/Confusion_Matrix', fig, epoch)
    plt.close()


def main():
    # 加载配置文件
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 设置随机种子
    set_seed(config['seed'])
    
    # 确定训练设备
    device = get_device(config['device'])
    print(f"使用设备: {device}")
    
    # 创建实验输出目录
    output_dir = Path(config['output_dir']) / config['experiment_name']
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logs_dir = output_dir / 'logs'
    checkpoints_dir = output_dir / 'checkpoints'
    logs_dir.mkdir(exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(output_dir / 'training.log'),
            logging.StreamHandler()
        ]
    )
    
    # 初始化TensorBoard写入器
    writer = SummaryWriter(logs_dir)
    
    # 创建数据变换
    transform = PreprocessTransform()
    
    # 创建数据集（这里假设数据已经分为train/val子目录）
    train_dataset = InterferenceDataset(
        Path(config['data_path']) / 'train',
        transform=transform,
        split='train'
    )
    
    val_dataset = InterferenceDataset(
        Path(config['data_path']) / 'val',
        transform=transform,
        split='val'
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # 创建模型
    model = create_model(num_classes=4)
    model = model.to(device)
    
    # 设置损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 设置优化器（判别性学习率）
    if config['use_discriminative_lr']:
        param_groups = setup_discriminative_lr(
            model,
            base_lr=config['learning_rates']['classifier'],
            layer_lr_decay=config['learning_rates']['layer_decay']
        )
        optimizer = optim.Adam(param_groups, weight_decay=config['weight_decay'])
    else:
        optimizer = optim.Adam(
            model.parameters(),
            lr=config['learning_rates']['base'],
            weight_decay=config['weight_decay']
        )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # 训练循环
    best_val_acc = 0.0
    best_model_path = checkpoints_dir / 'best_model.pth'
    
    logging.info("开始训练...")
    logging.info(f"训练集大小: {len(train_dataset)}")
    logging.info(f"验证集大小: {len(val_dataset)}")
    logging.info(f"批次大小: {config['batch_size']}")
    logging.info(f"训练轮数: {config['epochs']}")
    
    for epoch in range(1, config['epochs'] + 1):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch}/{config['epochs']}")
        print('='*50)
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # 验证
        val_loss, val_acc, cm = validate_epoch(
            model, val_loader, criterion, device, epoch
        )
        
        # 更新学习率
        scheduler.step(val_loss)
        
        # 记录到TensorBoard
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        
        # 记录混淆矩阵
        save_confusion_matrix_to_tensorboard(writer, cm, epoch)
        
        # 记录学习率
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'config': config
            }, best_model_path)
            
            # 保存混淆矩阵图像
            plot_confusion_matrix(
                cm, 
                ['Class 0', 'Class 1', 'Class 2', 'Class 3'],
                checkpoints_dir / f'confusion_matrix_epoch_{epoch}.png'
            )
            
            logging.info(f"新的最佳模型已保存! 验证准确率: {best_val_acc:.2f}%")
        
        # 定期保存检查点
        if epoch % 10 == 0:
            checkpoint_path = checkpoints_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'config': config
            }, checkpoint_path)
        
        logging.info(f"Epoch {epoch} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # 训练完成
    writer.close()
    logging.info("训练完成!")
    logging.info(f"最佳验证准确率: {best_val_acc:.2f}%")
    logging.info(f"最佳模型保存位置: {best_model_path}")
    
    # 保存训练配置
    with open(output_dir / 'final_config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


if __name__ == '__main__':
    main()