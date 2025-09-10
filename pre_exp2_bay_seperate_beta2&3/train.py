import os
import yaml
import random
import re  # NEW: 导入正则表达式库
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
    MODIFIED: 自定义数据集类，用于加载.npy格式的干涉条纹图像
    - 能够动态发现beta值并将其作为类别
    """
    def __init__(self, data_path, transform=None, split='train'):
        self.data_path = Path(data_path)
        self.transform = transform
        self.split = split
        
        # 获取所有.npy文件
        self.file_paths = sorted(list(self.data_path.glob('**/*.npy'))) # 排序以保证一致性
        
        # MODIFIED: 动态发现类别 (beta值) 并创建映射
        self.labels = []
        self.valid_files = []
        
        # 从文件名中提取所有beta值
        all_betas_str = []
        for file_path in self.file_paths:
            filename = file_path.stem
            # 使用正则表达式匹配 'betaX_Y.YYYY' 格式
            match = re.search(r"beta3_([\d.]+)", filename)
            if match:
                all_betas_str.append(match.group(1))
        
        # 获取唯一的、排序后的beta值作为我们的类别定义
        unique_betas = sorted(list(set(float(b) for b in all_betas_str)))
        self.sorted_betas = unique_betas
        self.num_classes = len(self.sorted_betas)
        self.beta_to_idx = {beta: i for i, beta in enumerate(self.sorted_betas)}
        
        print(f"在 {split} 集中动态发现 {self.num_classes} 个类别 (beta 值):")
        print(self.sorted_betas)

        # 再次遍历文件以分配标签
        for file_path in self.file_paths:
            filename = file_path.stem
            match = re.search(r"beta3_([\d.]+)", filename)
            if match:
                beta_val = float(match.group(1))
                label = self.beta_to_idx[beta_val]
                self.labels.append(label)
                self.valid_files.append(file_path)
        
        print(f"{split}集加载完成: {len(self.valid_files)} 个文件")
        print(f"类别分布 (整数标签): {np.bincount(self.labels)}")
    
    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        file_path = self.valid_files[idx]
        image = np.load(file_path)
        label = self.labels[idx]
        
        # 假设图像尺寸统一，如果需要可以取消注释
        # if image.shape != (645, 645):
        #     raise ValueError(f"图像尺寸错误: {image.shape}, 期望: (645, 645)")
        
        image = torch.from_numpy(image).float().unsqueeze(0)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# PreprocessTransform 和其他辅助函数保持不变
class PreprocessTransform:
    def __init__(self, resize_dim=224):
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.resize = transforms.Resize((resize_dim, resize_dim))
    
    def __call__(self, image):
        image = self.resize(image)
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

def create_model(num_classes):
    model = models.resnet34(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

# setup_discriminative_lr, train_epoch, validate_epoch, 等函数保持不变
# ... (此处省略与您提供代码中完全相同的部分以节约篇幅) ...
# train_epoch, validate_epoch, plot_confusion_matrix, etc. are identical to your provided script.
def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch} - Training')
    for i, (inputs, labels) in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix({'Loss': f'{running_loss / (i + 1):.4f}', 'Acc': f'{100.*correct/total:.2f}%'})
    return running_loss / len(train_loader), 100. * correct / total

def validate_epoch(model, val_loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_predicted, all_labels = [], []
    with torch.no_grad():
        pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f'Epoch {epoch} - Validation')
        for i, (inputs, labels) in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_predicted.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            pbar.set_postfix({'Loss': f'{running_loss / (i + 1):.4f}', 'Acc': f'{100.*correct/total:.2f}%'})
    cm = confusion_matrix(all_labels, all_predicted)
    return running_loss / len(val_loader), 100. * correct / total, cm

def plot_confusion_matrix(cm, class_names, output_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def save_confusion_matrix_to_tensorboard(writer, cm, class_names, epoch):
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    writer.add_figure('Validation/Confusion_Matrix', fig, epoch)
    plt.close()

def setup_discriminative_lr(model, base_lr, layer_lr_decay):
    # identical to your provided script
    early_layers = list(model.conv1.parameters()) + list(model.bn1.parameters()) + list(model.layer1.parameters()) + list(model.layer2.parameters())
    mid_layers = list(model.layer3.parameters())
    high_layers = list(model.layer4.parameters())
    classifier = list(model.fc.parameters())
    return [
        {'params': early_layers, 'lr': base_lr * (layer_lr_decay ** 3)},
        {'params': mid_layers, 'lr': base_lr * (layer_lr_decay ** 2)},
        {'params': high_layers, 'lr': base_lr * layer_lr_decay},
        {'params': classifier, 'lr': base_lr}
    ]

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
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                        handlers=[logging.FileHandler(output_dir / 'training.log'), logging.StreamHandler()])
    
    writer = SummaryWriter(str(logs_dir))
    
    transform = PreprocessTransform()
    
    train_dataset = InterferenceDataset(Path(config['data_path']) / 'train', transform=transform, split='train')
    val_dataset = InterferenceDataset(Path(config['data_path']) / 'val', transform=transform, split='val')
    
    # MODIFIED: 动态获取类别信息
    num_classes = train_dataset.num_classes
    sorted_betas = train_dataset.sorted_betas
    class_names = [f'{b:.4f}' for b in sorted_betas] # 用于混淆矩阵绘图
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                              num_workers=config['num_workers'], pin_memory=device.type == 'cuda')
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False,
                            num_workers=config['num_workers'], pin_memory=device.type == 'cuda')
    
    model = create_model(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    
    if config.get('use_discriminative_lr', False):
        param_groups = setup_discriminative_lr(model, base_lr=config['learning_rates']['classifier'],
                                               layer_lr_decay=config['learning_rates']['layer_decay'])
        optimizer = optim.Adam(param_groups, weight_decay=config['weight_decay'])
    else:
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rates']['base'], weight_decay=config['weight_decay'])
        
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    best_val_acc = 0.0
    best_model_path = checkpoints_dir / 'best_model.pth'
    
    logging.info("开始训练...")
    # ... (与您代码相同的训练循环) ...
    for epoch in range(1, config['epochs'] + 1):
        logging.info(f"\n{'='*50}\nEpoch {epoch}/{config['epochs']}\n{'='*50}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc, cm = validate_epoch(model, val_loader, criterion, device, epoch)
        
        scheduler.step(val_loss)
        
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        save_confusion_matrix_to_tensorboard(writer, cm, class_names, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # MODIFIED: 在保存检查点时加入 sorted_betas
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'sorted_betas': sorted_betas, # <-- KEY ADDITION
                'config': config
            }, best_model_path)
            
            plot_confusion_matrix(cm, class_names, checkpoints_dir / f'confusion_matrix_epoch_{epoch}.png')
            logging.info(f"新的最佳模型已保存! 验证准确率: {best_val_acc:.2f}%")

        if epoch % 10 == 0:
            torch.save({ 'epoch': epoch, 'model_state_dict': model.state_dict() }, 
                       checkpoints_dir / f'checkpoint_epoch_{epoch}.pth')

        logging.info(f"Epoch {epoch} Results -> Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                     f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

    writer.close()
    logging.info(f"训练完成! 最佳验证准确率: {best_val_acc:.2f}% at {best_model_path}")

if __name__ == '__main__':
    main()