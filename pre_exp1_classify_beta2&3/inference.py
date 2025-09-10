class InferenceDataset(Dataset):
    """推理专用数据集类"""
    def __init__(self, data_path, transform=None, has_labels=True, num_classes=4):
        self.data_path = Path(data_path)
        self.transform = transform
        self.has_labels = has_labels
        self.num_classes = num_classes
        
        # 获取所有.npy文件
        self.file_paths = list(self.data_path.glob('**/*.npy'))
        self.file_paths.sort()  # 确保顺序一致
        
        if self.has_labels:
            # 如果有标签，从文件名提取
            self.labels = []
            self.valid_files = []
            
            for file_path in self.file_paths:
                try:
                    label = self._extract_class_from_filename(file_path.stem)
                    if label is not None and 0 <= label < self.num_classes:
                        self.labels.append(label)
                        self.valid_files.append(file_path)
                    elif label is not None:
                        # 类别超出范围，设置为-1
                import torch
import numpy as np
import yaml
from pathlib import Path
import argparse
from train import create_model, PreprocessTransform, get_device
from utils import load_checkpoint, evaluate_model, generate_prediction_samples
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class InferenceDataset(Dataset):
    """推理专用数据集类"""
    def __init__(self, data_path, transform=None, has_labels=True):
        self.data_path = Path(data_path)
        self.transform = transform
        self.has_labels = has_labels
        
        # 获取所有.npy文件
        self.file_paths = list(self.data_path.glob('**/*.npy'))
        self.file_paths.sort()  # 确保顺序一致
        
        if self.has_labels:
            # 如果有标签，从文件名提取
            self.labels = []
            self.valid_files = []
            
            for file_path in self.file_paths:
                try:
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
                        # 如果无法从文件名识别类别，设置为-1
                        label = -1
                    
                    self.labels.append(label)
                    self.valid_files.append(file_path)
                except Exception as e:
                    print(f"跳过文件 {file_path}: {e}")
                    continue
        else:
            # 无标签推理
            self.valid_files = self.file_paths
            self.labels = [-1] * len(self.valid_files)
        
        print(f"推理数据集加载完成: {len(self.valid_files)} 个文件")
    
    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        file_path = self.valid_files[idx]
        image = np.load(file_path)
        label = self.labels[idx]
        
        # 转换为tensor
        image = torch.from_numpy(image).float().unsqueeze(0)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, str(file_path)


def single_image_inference(model, image_path, device, transform):
    """单张图像推理"""
    model.eval()
    
    # 加载图像
    image = np.load(image_path)
    image_tensor = torch.from_numpy(image).float().unsqueeze(0)  # (1, 645, 645)
    
    if transform:
        image_tensor = transform(image_tensor)
    
    image_tensor = image_tensor.unsqueeze(0).to(device)  # (1, 3, 224, 224)
    
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = output.argmax(1).item()
        confidence = probabilities.max().item()
    
    return predicted_class, confidence, probabilities.cpu().numpy()[0]


def batch_inference(model, data_loader, device, save_results=True, output_file=None):
    """批量推理"""
    model.eval()
    
    all_predictions = []
    all_confidences = []
    all_filenames = []
    all_true_labels = []
    
    with torch.no_grad():
        for images, labels, filenames in data_loader:
            images = images.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_classes = outputs.argmax(1)
            confidences = probabilities.max(1)[0]
            
            # 收集结果
            all_predictions.extend(predicted_classes.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            all_filenames.extend([Path(f).name for f in filenames])
            all_true_labels.extend(labels.numpy())
    
    # 创建结果字典
    results = {
        'filename': all_filenames,
        'predicted_class': all_predictions,
        'confidence': all_confidences,
        'true_label': all_true_labels
    }
    
    # 计算准确率（如果有真实标签）
    if any(label != -1 for label in all_true_labels):
        valid_indices = [i for i, label in enumerate(all_true_labels) if label != -1]
        if valid_indices:
            valid_predictions = [all_predictions[i] for i in valid_indices]
            valid_labels = [all_true_labels[i] for i in valid_indices]
            accuracy = np.mean(np.array(valid_predictions) == np.array(valid_labels))
            print(f"推理准确率: {accuracy:.4f}")
    
    # 保存结果
    if save_results and output_file:
        import pandas as pd
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"推理结果已保存到: {output_file}")
    
    return results


def analyze_predictions(results):
    """分析预测结果"""
    predictions = results['predicted_class']
    confidences = results['confidence']
    
    # 类别分布
    unique, counts = np.unique(predictions, return_counts=True)
    print("\n预测类别分布:")
    for class_id, count in zip(unique, counts):
        print(f"  Class {class_id}: {count} 个样本")
    
    # 置信度统计
    print(f"\n置信度统计:")
    print(f"  平均置信度: {np.mean(confidences):.4f}")
    print(f"  置信度标准差: {np.std(confidences):.4f}")
    print(f"  最低置信度: {np.min(confidences):.4f}")
    print(f"  最高置信度: {np.max(confidences):.4f}")
    
    # 低置信度样本
    low_confidence_threshold = 0.7
    low_conf_indices = np.where(np.array(confidences) < low_confidence_threshold)[0]
    if len(low_conf_indices) > 0:
        print(f"\n低置信度样本 (< {low_confidence_threshold}):")
        for idx in low_conf_indices[:5]:  # 只显示前5个
            filename = results['filename'][idx]
            pred_class = results['predicted_class'][idx]
            confidence = results['confidence'][idx]
            print(f"  {filename}: Class {pred_class}, 置信度 {confidence:.4f}")


def main():
    parser = argparse.ArgumentParser(description='干涉条纹图像分类推理脚本')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型检查点路径')
    parser.add_argument('--data_path', type=str, required=True,
                        help='推理数据路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出结果文件路径')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批处理大小')
    parser.add_argument('--single_image', type=str, default=None,
                        help='单张图像推理路径')
    parser.add_argument('--generate_samples', action='store_true',
                        help='生成预测样本图像')
    parser.add_argument('--has_labels', action='store_true', default=False,
                        help='数据是否包含标签（用于计算准确率）')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 确定设备
    device = get_device(config['device'])
    print(f"使用设备: {device}")
    
    # 创建模型
    model = create_model(num_classes=4)
    model = model.to(device)
    
    # 加载检查点
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"检查点文件不存在: {checkpoint_path}")
    
    load_checkpoint(checkpoint_path, model)
    print("模型加载完成")
    
    # 创建预处理变换
    transform = PreprocessTransform()
    
    if args.single_image:
        # 单张图像推理
        print(f"\n对单张图像进行推理: {args.single_image}")
        predicted_class, confidence, probabilities = single_image_inference(
            model, args.single_image, device, transform
        )
        
        print(f"预测结果:")
        print(f"  预测类别: Class {predicted_class}")
        print(f"  置信度: {confidence:.4f}")
        print(f"  各类别概率:")
        for i, prob in enumerate(probabilities):
            print(f"    Class {i}: {prob:.4f}")
    
    else:
        # 批量推理
        print(f"\n对目录进行批量推理: {args.data_path}")
        
        # 创建数据集和加载器
        dataset = InferenceDataset(
            args.data_path,
            transform=transform,
            has_labels=args.has_labels
        )
        
        data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=config.get('num_workers', 4)
        )
        
        # 执行推理
        if args.output is None:
            timestamp = Path(args.checkpoint).stem
            args.output = f"inference_results_{timestamp}.csv"
        
        results = batch_inference(
            model, data_loader, device,
            save_results=True, output_file=args.output
        )
        
        # 分析结果
        analyze_predictions(results)
        
        # 生成预测样本图像
        if args.generate_samples:
            samples_path = Path(args.output).parent / f"prediction_samples_{Path(args.checkpoint).stem}.png"
            print(f"\n生成预测样本图像: {samples_path}")
            generate_prediction_samples(
                model, dataset, device, 
                num_samples=16, save_path=samples_path
            )
        
        # 如果有标签，进行详细评估
        if args.has_labels and any(label != -1 for label in results['true_label']):
            print("\n执行详细评估...")
            
            # 创建只包含有标签样本的数据加载器
            labeled_indices = [i for i, label in enumerate(results['true_label']) if label != -1]
            if labeled_indices:
                from torch.utils.data import Subset
                labeled_dataset = Subset(dataset, labeled_indices)
                labeled_loader = DataLoader(
                    labeled_dataset,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=config.get('num_workers', 4)
                )
                
                eval_results = evaluate_model(
                    model, labeled_loader, device,
                    num_classes=config['num_classes'],
                    class_names=[f'Class {i}' for i in range(config['num_classes'])]
                )


if __name__ == '__main__':
    main()