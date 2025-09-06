import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import yaml
from sklearn.metrics import classification_report, precision_recall_fscore_support
import seaborn as sns


def load_config(config_path='config.yaml'):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def create_data_splits(data_root, train_ratio=0.8, val_ratio=0.2):
    """
    创建数据集分割（如果数据还没有分割的话）
    """
    data_root = Path(data_root)
    
    # 获取所有.npy文件
    all_files = list(data_root.glob('**/*.npy'))
    
    # 按类别分组
    class_files = {0: [], 1: [], 2: [], 3: []}
    
    for file_path in all_files:
        filename = file_path.stem.lower()
        if 'class0' in filename:
            class_files[0].append(file_path)
        elif 'class1' in filename:
            class_files[1].append(file_path)
        elif 'class2' in filename:
            class_files[2].append(file_path)
        elif 'class3' in filename:
            class_files[3].append(file_path)
    
    # 创建train和val目录
    train_dir = data_root / 'train'
    val_dir = data_root / 'val'
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    # 为每个类别创建分割
    for class_id, files in class_files.items():
        np.random.shuffle(files)
        
        n_train = int(len(files) * train_ratio)
        train_files = files[:n_train]
        val_files = files[n_train:]
        
        # 创建符号链接或复制文件
        for file_path in train_files:
            target_path = train_dir / file_path.name
            if not target_path.exists():
                target_path.symlink_to(file_path.absolute())
        
        for file_path in val_files:
            target_path = val_dir / file_path.name
            if not target_path.exists():
                target_path.symlink_to(file_path.absolute())
        
        print(f"Class {class_id}: {len(train_files)} train, {len(val_files)} val")


def visualize_samples(dataset, num_samples=8, save_path=None):
    """可视化数据集样本"""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    # 随机选择样本
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for i, idx in enumerate(indices):
        image, label = dataset[idx]
        
        # 如果是经过预处理的3通道图像，取第一个通道
        if image.shape[0] == 3:
            image_np = image[0].numpy()
        else:
            image_np = image.squeeze().numpy()
        
        axes[i].imshow(image_np, cmap='gray')
        axes[i].set_title(f'Class {label}')
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def analyze_dataset(data_path):
    """分析数据集统计信息"""
    data_path = Path(data_path)
    
    # 统计各个子目录的文件数量
    for split in ['train', 'val', 'test']:
        split_path = data_path / split
        if split_path.exists():
            files = list(split_path.glob('*.npy'))
            
            # 统计类别分布
            class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
            
            for file_path in files:
                filename = file_path.stem.lower()
                if 'class0' in filename:
                    class_counts[0] += 1
                elif 'class1' in filename:
                    class_counts[1] += 1
                elif 'class2' in filename:
                    class_counts[2] += 1
                elif 'class3' in filename:
                    class_counts[3] += 1
            
            print(f"\n{split.upper()} 集统计:")
            print(f"总文件数: {len(files)}")
            for class_id, count in class_counts.items():
                print(f"  Class {class_id}: {count} 个文件")


def load_checkpoint(checkpoint_path, model, optimizer=None):
    """加载模型检查点"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    best_val_acc = checkpoint.get('best_val_acc', 0.0)
    
    print(f"检查点加载成功: epoch {epoch}, 最佳验证准确率 {best_val_acc:.2f}%")
    
    return epoch, best_val_acc


def evaluate_model(model, test_loader, device, class_names=None):
    """在测试集上评估模型"""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算各种指标
    accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average='macro'
    )
    
    # 生成分类报告
    if class_names is None:
        class_names = [f'Class {i}' for i in range(4)]
    
    report = classification_report(
        all_labels, all_predictions,
        target_names=class_names,
        digits=4
    )
    
    print("测试集评估结果:")
    print(f"准确率: {accuracy:.4f}")
    print(f"宏平均精确率: {precision:.4f}")
    print(f"宏平均召回率: {recall:.4f}")
    print(f"宏平均F1分数: {f1:.4f}")
    print("\n详细分类报告:")
    print(report)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_predictions,
        'labels': all_labels,
        'report': report
    }


def plot_training_history(log_dir, save_path=None):
    """绘制训练历史曲线"""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    
    # 读取TensorBoard日志
    ea = EventAccumulator(str(log_dir))
    ea.Reload()
    
    # 获取标量数据
    train_loss = ea.Scalars('Loss/Train')
    val_loss = ea.Scalars('Loss/Validation')
    train_acc = ea.Scalars('Accuracy/Train')
    val_acc = ea.Scalars('Accuracy/Validation')
    
    # 提取数据
    epochs = [x.step for x in train_loss]
    train_loss_values = [x.value for x in train_loss]
    val_loss_values = [x.value for x in val_loss]
    train_acc_values = [x.value for x in train_acc]
    val_acc_values = [x.value for x in val_acc]
    
    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # 损失曲线
    ax1.plot(epochs, train_loss_values, label='训练损失', color='blue')
    ax1.plot(epochs, val_loss_values, label='验证损失', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('训练和验证损失')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
    ax2.plot(epochs, train_acc_values, label='训练准确率', color='blue')
    ax2.plot(epochs, val_acc_values, label='验证准确率', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('训练和验证准确率')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def calculate_model_size(model):
    """计算模型参数数量"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型参数统计:")
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数数量: {trainable_params:,}")
    print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (假设float32)")
    
    return total_params, trainable_params


def check_data_integrity(data_path):
    """检查数据完整性"""
    data_path = Path(data_path)
    issues = []
    
    for split in ['train', 'val']:
        split_path = data_path / split
        if not split_path.exists():
            issues.append(f"缺少 {split} 目录")
            continue
        
        files = list(split_path.glob('*.npy'))
        if len(files) == 0:
            issues.append(f"{split} 目录为空")
            continue
        
        # 检查每个文件
        for file_path in files:
            try:
                data = np.load(file_path)
                if data.shape != (645, 645):
                    issues.append(f"文件 {file_path} 尺寸错误: {data.shape}")
                if data.dtype not in [np.float32, np.float64]:
                    issues.append(f"文件 {file_path} 数据类型错误: {data.dtype}")
            except Exception as e:
                issues.append(f"无法读取文件 {file_path}: {str(e)}")
    
    if issues:
        print("发现以下数据问题:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("数据完整性检查通过!")
        return True


def generate_prediction_samples(model, dataset, device, num_samples=16, save_path=None):
    """生成预测样本图像"""
    model.eval()
    
    # 随机选择样本
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, true_label = dataset[idx]
            
            # 预测
            image_tensor = image.unsqueeze(0).to(device)
            output = model(image_tensor)
            predicted_label = output.argmax(1).item()
            confidence = torch.softmax(output, 1).max().item()
            
            # 可视化
            if image.shape[0] == 3:
                image_np = image[0].numpy()
            else:
                image_np = image.squeeze().numpy()
            
            axes[i].imshow(image_np, cmap='gray')
            
            # 设置标题颜色（正确预测为绿色，错误为红色）
            color = 'green' if predicted_label == true_label else 'red'
            axes[i].set_title(
                f'True: {true_label}, Pred: {predicted_label}\n'
                f'Confidence: {confidence:.3f}',
                color=color
            )
            axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def create_submission_file(model, test_loader, device, output_path):
    """创建提交文件"""
    model.eval()
    predictions = []
    filenames = []
    
    with torch.no_grad():
        for batch_idx, (inputs, _) in enumerate(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # 假设测试集有文件名信息，这里简化处理
            batch_predictions = predicted.cpu().numpy()
            predictions.extend(batch_predictions)
            
            # 生成文件名（实际应用中应该从数据集获取）
            for i in range(len(batch_predictions)):
                filenames.append(f"test_{batch_idx * test_loader.batch_size + i:04d}.npy")
    
    # 创建提交文件
    submission_data = {
        'filename': filenames,
        'prediction': predictions
    }
    
    import pandas as pd
    df = pd.DataFrame(submission_data)
    df.to_csv(output_path, index=False)
    print(f"提交文件已保存到: {output_path}")


if __name__ == '__main__':
    # 示例用法
    print("工具脚本加载完成!")
    
    # 分析数据集
    # analyze_dataset('data/interference_images')
    
    # 检查数据完整性
    # check_data_integrity('data/interference_images')