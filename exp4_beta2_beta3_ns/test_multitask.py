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
from sklearn.metrics import r2_score, confusion_matrix, mean_squared_error, mean_absolute_error, classification_report
import sys
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

# 导入训练程序中的类
from train2 import (
    MultiTaskInterferenceDataset, 
    MultiTaskResNetMLP, 
    CustomLogger,
    create_model, 
    set_seed, 
    get_device,
    DataPreprocessor,
    CenterCropTensor,
    LogNormalization
)


class GradCAM:
    """Grad-CAM实现，用于生成类激活图"""
    
    def __init__(self, model, target_layers, task_type='classification'):
        """
        初始化Grad-CAM
        
        Args:
            model: 训练好的模型
            target_layers: 目标层名称列表
            task_type: 任务类型，'classification' 或 'regression'
        """
        self.model = model
        self.target_layers = target_layers
        self.task_type = task_type
        self.gradients = {}
        self.activations = {}
        self.hooks = []
        
        self._register_hooks()
    
    def _register_hooks(self):
        """注册前向和反向钩子"""
        def forward_hook(name):
            def hook(module, input, output):
                self.activations[name] = output.detach()
            return hook
        
        def backward_hook(name):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    self.gradients[name] = grad_output[0].detach()
            return hook
    
        # 为目标层注册钩子 - 只注册完全匹配的层
        registered_layers = set()  # 避免重复注册
        
        for name, module in self.model.named_modules():
            # 精确匹配目标层名称
            for target_layer in self.target_layers:
                if name.endswith(target_layer) and name not in registered_layers:
                    # 注册前向钩子
                    handle_f = module.register_forward_hook(forward_hook(name))
                    # 注册反向钩子
                    handle_b = module.register_full_backward_hook(backward_hook(name))
                    self.hooks.extend([handle_f, handle_b])
                    registered_layers.add(name)
                    print(f"为层 {name} 注册了钩子")
                    break  # 找到匹配的层后跳出内循环
    
    def generate_cam(self, input_tensor, target_class=None, regression_dim=None):
        """
        生成类激活图
        
        Args:
            input_tensor: 输入张量
            target_class: 目标类别（分类任务）
            regression_dim: 回归维度索引（回归任务）
            
        Returns:
            cam: 类激活图
        """
        # 清空之前的激活和梯度
        self.gradients.clear()
        self.activations.clear()
        
        # 前向传播
        self.model.eval()
        input_tensor.requires_grad_(True)  # 确保输入需要梯度
        reg_output, cls_output = self.model(input_tensor)
        
        # 根据任务类型选择目标
        if self.task_type == 'classification':
            if target_class is None:
                target_class = torch.argmax(cls_output, dim=1)
            score = cls_output[0, target_class]
        elif self.task_type == 'regression':
            if regression_dim is None:
                regression_dim = 0  # 默认使用第一个维度（beta2）
            score = reg_output[0, regression_dim]
        else:
            raise ValueError(f"不支持的任务类型: {self.task_type}")
        
        # 反向传播
        self.model.zero_grad()
        score.backward(retain_graph=False)
        
        # 计算Grad-CAM
        cams = []
        for name, module in self.model.named_modules():
            if any(target in name for target in self.target_layers):
                if name in self.gradients and name in self.activations:
                    gradients = self.gradients[name]
                    activations = self.activations[name]
                    
                    # 计算权重（全局平均池化）
                    weights = torch.mean(gradients, dim=[2, 3], keepdim=True)
                    
                    # 加权求和
                    cam = torch.sum(weights * activations, dim=1, keepdim=True)
                    
                    # ReLU激活
                    cam = F.relu(cam)
                    
                    # 归一化到[0, 1]
                    cam = cam.squeeze()
                    if cam.numel() > 1:  # 确保不是标量
                        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
                    
                    cams.append(cam)
                    break  # 只使用第一个匹配的层
        
        return cams[0] if cams else None
    
    def cleanup(self):
        """清理钩子"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.gradients.clear()
        self.activations.clear()


class FeatureExtractor:
    """特征提取器，用于保存ResNet特征"""
    
    def __init__(self, model):
        self.model = model
        self.features = None
        self.hook = None
        self._register_hook()
    
    def _register_hook(self):
        """注册前向钩子来捕获特征"""
        def hook(module, input, output):
            self.features = output.detach().cpu().numpy()
        
        # 在特征投射层之前捕获特征
        self.hook = self.model.feature_extractor.features.register_forward_hook(hook)
    
    def extract_features(self, input_tensor):
        """提取特征"""
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor)
        return self.features.copy() if self.features is not None else None
    
    def cleanup(self):
        """清理钩子"""
        if self.hook:
            self.hook.remove()


def load_model_from_checkpoint(checkpoint_path, config, device):
    """从checkpoint加载模型"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型checkpoint不存在: {checkpoint_path}")
    
    # 加载checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 创建模型
    model = create_model(config, device=device)
    model.to(device)
    
    # 加载模型权重
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"成功加载模型权重，来自epoch {checkpoint['epoch']}")
    print(f"最佳验证损失: {checkpoint.get('best_total_loss', 'N/A')}")
    
    return model, checkpoint


def test_model(model, test_loader, device, config):
    """测试模型并收集预测结果"""
    model.eval()
    
    all_reg_preds, all_reg_labels = [], []
    all_cls_preds, all_cls_labels = [], []
    all_file_paths = []
    all_processed_images = []
    all_features = []
    
    # 设置特征提取器
    feature_extractor = None
    if config['test_config']['save_features']['enabled']:
        feature_extractor = FeatureExtractor(model)
    
    print("开始测试模型...")
    with torch.no_grad():
        for batch_idx, (inputs, reg_labels, cls_labels, file_paths, processed_images) in enumerate(tqdm(test_loader)):
            inputs = inputs.to(device)
            reg_labels = reg_labels.to(device)
            cls_labels = cls_labels.to(device)
            
            # 前向传播
            reg_outputs, cls_outputs = model(inputs)
            
            # 收集预测结果
            all_reg_preds.extend(reg_outputs.cpu().numpy())
            all_reg_labels.extend(reg_labels.cpu().numpy())
            
            _, predicted = cls_outputs.max(1)
            all_cls_preds.extend(predicted.cpu().numpy())
            all_cls_labels.extend(cls_labels.cpu().numpy())
            
            # 收集文件路径
            all_file_paths.extend(file_paths)
            
            # 收集预处理后的图像
            if config['test_config']['save_features']['include_processed_images']:
                all_processed_images.extend(processed_images.cpu().numpy())
            
            # 提取特征
            if feature_extractor is not None:
                features = feature_extractor.extract_features(inputs)
                if features is not None:
                    all_features.extend(features)
    
    if feature_extractor is not None:
        feature_extractor.cleanup()
    
    return {
        'reg_predictions': np.array(all_reg_preds),
        'reg_labels': np.array(all_reg_labels),
        'cls_predictions': np.array(all_cls_preds),
        'cls_labels': np.array(all_cls_labels),
        'file_paths': all_file_paths,
        'processed_images': np.array(all_processed_images) if all_processed_images else None,
        'features': np.array(all_features) if all_features else None
    }


def evaluate_results(results):
    """评估测试结果"""
    reg_preds = results['reg_predictions']
    reg_labels = results['reg_labels']
    cls_preds = results['cls_predictions']
    cls_labels = results['cls_labels']
    
    # 回归评估指标
    reg_mse = mean_squared_error(reg_labels, reg_preds)
    reg_mae = mean_absolute_error(reg_labels, reg_preds)
    
    mse_beta2 = mean_squared_error(reg_labels[:, 0], reg_preds[:, 0])
    mse_beta3 = mean_squared_error(reg_labels[:, 1], reg_preds[:, 1])
    mae_beta2 = mean_absolute_error(reg_labels[:, 0], reg_preds[:, 0])
    mae_beta3 = mean_absolute_error(reg_labels[:, 1], reg_preds[:, 1])
    
    r2_beta2 = r2_score(reg_labels[:, 0], reg_preds[:, 0])
    r2_beta3 = r2_score(reg_labels[:, 1], reg_preds[:, 1])
    
    # 分类评估指标
    test_acc = 100. * np.sum(cls_preds == cls_labels) / len(cls_labels)
    cm = confusion_matrix(cls_labels, cls_preds)
    
    # 分类报告
    cls_report = classification_report(cls_labels, cls_preds, target_names=['Class 0', 'Class 1', 'Class 2'])
    
    metrics = {
        'regression_mse': reg_mse,
        'regression_mae': reg_mae,
        'mse_beta2': mse_beta2, 'mse_beta3': mse_beta3,
        'mae_beta2': mae_beta2, 'mae_beta3': mae_beta3,
        'r2_beta2': r2_beta2, 'r2_beta3': r2_beta3,
        'classification_accuracy': test_acc,
        'confusion_matrix': cm,
        'classification_report': cls_report
    }
    
    return metrics


def plot_test_results(results, metrics, output_path):
    """绘制测试结果"""
    reg_preds = results['reg_predictions']
    reg_labels = results['reg_labels']
    cls_preds = results['cls_predictions']
    cls_labels = results['cls_labels']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 第一行：回归结果
    # Beta2 预测 vs 真实值
    axes[0, 0].scatter(reg_labels[:, 0], reg_preds[:, 0], alpha=0.6)
    axes[0, 0].plot([reg_labels[:, 0].min(), reg_labels[:, 0].max()], 
                    [reg_labels[:, 0].min(), reg_labels[:, 0].max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('True Beta2')
    axes[0, 0].set_ylabel('Predicted Beta2')
    axes[0, 0].set_title('Beta2 Predictions vs True Values')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].text(0.05, 0.95, f'MSE: {metrics["mse_beta2"]:.6f}\nR²: {metrics["r2_beta2"]:.6f}', 
                   transform=axes[0, 0].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Beta3 预测 vs 真实值
    axes[0, 1].scatter(reg_labels[:, 1], reg_preds[:, 1], alpha=0.6)
    axes[0, 1].plot([reg_labels[:, 1].min(), reg_labels[:, 1].max()], 
                    [reg_labels[:, 1].min(), reg_labels[:, 1].max()], 'r--', lw=2)
    axes[0, 1].set_xlabel('True Beta3')
    axes[0, 1].set_ylabel('Predicted Beta3')
    axes[0, 1].set_title('Beta3 Predictions vs True Values')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].text(0.05, 0.95, f'MSE: {metrics["mse_beta3"]:.6f}\nR²: {metrics["r2_beta3"]:.6f}', 
                   transform=axes[0, 1].transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2D散点图：(beta2, beta3)空间
    axes[0, 2].scatter(reg_labels[:, 0], reg_labels[:, 1], alpha=0.6, label='True', s=30)
    axes[0, 2].scatter(reg_preds[:, 0], reg_preds[:, 1], alpha=0.6, label='Predicted', s=30)
    axes[0, 2].set_xlabel('Beta2')
    axes[0, 2].set_ylabel('Beta3')
    axes[0, 2].set_title('Predictions in (Beta2, Beta3) Space')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 第二行：分类结果
    cm = metrics['confusion_matrix']
    class_names = ['Class 0', 'Class 1', 'Class 2']
    
    # 混淆矩阵
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[1, 0])
    axes[1, 0].set_title('Classification Confusion Matrix')
    axes[1, 0].set_xlabel('Predicted')
    axes[1, 0].set_ylabel('True')
    
    # 类别分布对比
    true_counts = np.bincount(cls_labels, minlength=3)
    pred_counts = np.bincount(cls_preds, minlength=3)
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
    
    # 测试性能总结
    axes[1, 2].text(0.5, 0.5, f'Test Results Summary:\n\n'
                               f'Classification Accuracy: {metrics["classification_accuracy"]:.2f}%\n\n'
                               f'Regression Metrics:\n'
                               f'  MSE: {metrics["regression_mse"]:.6f}\n'
                               f'  MAE: {metrics["regression_mae"]:.6f}\n\n'
                               f'Beta2: R² = {metrics["r2_beta2"]:.4f}\n'
                               f'Beta3: R² = {metrics["r2_beta3"]:.4f}\n\n'
                               f'Total Samples: {len(cls_labels)}', 
                   transform=axes[1, 2].transAxes, fontsize=12,
                   horizontalalignment='center', verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    axes[1, 2].set_title('Test Performance Summary')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def perform_gradcam_analysis(model, test_loader, device, config, output_dir):
    """执行Grad-CAM分析"""
    gradcam_config = config['interpretability']['gradcam']
    if not gradcam_config['enabled']:
        return
    
    print("开始Grad-CAM分析...")
    
    # 创建输出目录
    gradcam_dir = output_dir / gradcam_config['output_dir']
    gradcam_dir.mkdir(parents=True, exist_ok=True)
    
    # 为不同任务创建子目录
    if gradcam_config['task_specific']['regression']:
        (gradcam_dir / 'regression').mkdir(exist_ok=True)
    if gradcam_config['task_specific']['classification']:
        (gradcam_dir / 'classification').mkdir(exist_ok=True)
    
    # 初始化Grad-CAM - 创建一个实例用于两个任务
    target_layers = gradcam_config['target_layers']
    gradcam_analyzer = None
    
    # 只创建一个GradCAM实例
    if gradcam_config['task_specific']['classification'] or gradcam_config['task_specific']['regression']:
        gradcam_analyzer = GradCAM(model, target_layers, task_type='classification')  # 后面会动态改变任务类型
    
    # 处理样本
    num_samples = gradcam_config['num_samples']
    samples_processed = 0
    
    for batch_idx, (inputs, reg_labels, cls_labels, file_paths, processed_images) in enumerate(test_loader):
        if num_samples != -1 and samples_processed >= num_samples:
            break
        
        inputs = inputs.to(device)
        batch_size = inputs.size(0)
        
        for i in range(batch_size):
            if num_samples != -1 and samples_processed >= num_samples:
                break
            
            # 单个样本处理
            single_input = inputs[i:i+1]
            file_path = file_paths[i]
            processed_image = processed_images[i] if processed_images is not None else None
            
            # 获取文件名（用于保存）
            file_stem = Path(file_path).stem
            
            # 分类任务的Grad-CAM
            if gradcam_config['task_specific']['classification'] and gradcam_analyzer is not None:
                try:
                    gradcam_analyzer.task_type = 'classification'  # 设置任务类型
                    cam = gradcam_analyzer.generate_cam(single_input)
                    if cam is not None:
                        save_gradcam_visualization(
                            cam, processed_image, 
                            gradcam_dir / 'classification' / f'{file_stem}_cls_gradcam.png',
                            gradcam_config
                        )
                except Exception as e:
                    print(f"分类Grad-CAM处理失败 {file_stem}: {e}")
            
            # 回归任务的Grad-CAM（beta2和beta3）
            if gradcam_config['task_specific']['regression'] and gradcam_analyzer is not None:
                for reg_dim, reg_name in enumerate(['beta2', 'beta3']):
                    try:
                        gradcam_analyzer.task_type = 'regression'  # 设置任务类型
                        cam = gradcam_analyzer.generate_cam(single_input, regression_dim=reg_dim)
                        if cam is not None:
                            save_gradcam_visualization(
                                cam, processed_image, 
                                gradcam_dir / 'regression' / f'{file_stem}_{reg_name}_gradcam.png',
                                gradcam_config
                            )
                    except Exception as e:
                        print(f"回归Grad-CAM处理失败 {file_stem} {reg_name}: {e}")
            
            samples_processed += 1
            
            if samples_processed % 10 == 0:
                print(f"Grad-CAM分析进度: {samples_processed}/{num_samples if num_samples != -1 else '?'}")
    
    # 清理
    if gradcam_analyzer:
        gradcam_analyzer.cleanup()
    
    print(f"Grad-CAM分析完成，共处理 {samples_processed} 个样本")


def save_gradcam_visualization(cam, processed_image, output_path, config):
    """保存Grad-CAM可视化结果"""
    # 将CAM调整到原图尺寸
    if processed_image is not None:
        height, width = processed_image.shape[1], processed_image.shape[2]
        # 修改这一行：先移动到CPU再转换为numpy
        cam_resized = cv2.resize(cam.cpu().numpy(), (width, height))
    else:
        # 修改这一行：先移动到CPU再转换为numpy
        cam_resized = cam.cpu().numpy()
    
    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原图（如果有）
    if processed_image is not None:
        # 反标准化显示
        img_display = processed_image.numpy().transpose(2, 1, 0)
        # 简单的反标准化（假设使用了ImageNet标准化）
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_display = std * img_display + mean
        img_display = np.clip(img_display, 0, 1)
        
        axes[0].imshow(img_display)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
    else:
        axes[0].text(0.5, 0.5, 'No Image Available', ha='center', va='center')
        axes[0].axis('off')
    
    # Grad-CAM热力图
    colormap = config.get('colormap', 'jet')
    im = axes[1].imshow(cam_resized, cmap=colormap, alpha=1.0)
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1])
    
    # 叠加图
    if processed_image is not None and config.get('save_overlay', True):
        alpha = config.get('alpha', 0.4)
        # 将热力图转换为RGB
        cm = plt.get_cmap(colormap)
        heatmap_rgb = cm(cam_resized)[:, :, :3]
        
        # 叠加
        overlay = (1 - alpha) * img_display + alpha * heatmap_rgb
        overlay = np.clip(overlay, 0, 1)
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
    else:
        axes[2].text(0.5, 0.5, 'No Overlay Available', ha='center', va='center')
        axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_features_and_data(results, output_dir, config):
    """保存特征和数据"""
    if not config['test_config']['save_features']['enabled']:
        return
    
    features_dir = output_dir / config['test_config']['save_features']['output_dir']
    features_dir.mkdir(parents=True, exist_ok=True)
    
    print("保存特征和数据...")
    
    # 准备保存的数据
    save_data = {
        'reg_predictions': results['reg_predictions'],
        'reg_labels': results['reg_labels'],
        'cls_predictions': results['cls_predictions'],
        'cls_labels': results['cls_labels'],
        'file_paths': np.array(results['file_paths'], dtype='<U200')  # 字符串数组
    }
    
    # 添加预处理后的图像
    if config['test_config']['save_features']['include_processed_images'] and results['processed_images'] is not None:
        save_data['processed_images'] = results['processed_images']
    
    # 添加ResNet特征
    if results['features'] is not None:
        save_data['features'] = results['features']
    
    # 保存到npz文件
    features_file = features_dir / 'extracted_features.npz'
    np.savez_compressed(features_file, **save_data)
    
    print(f"特征和数据已保存到: {features_file}")
    print(f"包含内容:")
    print(f"  - 回归预测和标签: {results['reg_predictions'].shape}")
    print(f"  - 分类预测和标签: {results['cls_predictions'].shape}")
    print(f"  - 文件路径: {len(results['file_paths'])} 个")
    if results['processed_images'] is not None:
        print(f"  - 预处理图像: {results['processed_images'].shape}")
    if results['features'] is not None:
        print(f"  - ResNet特征: {results['features'].shape}")


class TestMultiTaskDataset(MultiTaskInterferenceDataset):
    """测试用的多任务数据集，扩展原数据集以返回额外信息"""
    
    def __getitem__(self, idx):
        # 调用父类方法
        image_tensor, regression_label, classification_label = super().__getitem__(idx)
        
        # 获取文件路径
        file_path = str(self.file_paths[idx])
        
        # 获取预处理前的图像（用于可视化）
        processed_image = image_tensor.clone()
        
        return image_tensor, regression_label, classification_label, file_path, processed_image


def main():
    """主测试函数"""
    # 加载配置
    with open('config_test.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 检查是否为测试模式
    if config.get('mode', 'train') != 'test':
        print("配置文件中mode不是'test'，请修改配置文件")
        return
    
    # 设置基本参数
    set_seed(config['seed'])
    device = get_device(config['device'])
    print(f"使用设备: {device}")
    
    # 创建输出目录
    output_dir = Path(config['output_dir']) / config['experiment_name']
    test_output_dir = output_dir / 'test_results'
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置日志
    log_file = test_output_dir / 'test.log'
    custom_logger = CustomLogger(log_file)
    sys.stdout = custom_logger
    
    try:
        print("="*60)
        print("开始多任务模型测试")
        print("="*60)
        
        # 加载模型
        checkpoint_path = config['test_config']['model_checkpoint']
        model, checkpoint = load_model_from_checkpoint(checkpoint_path, config, device)
        
        print("\n" + "="*80)
        print("               模型结构 - 所有层的名称               ")
        print("="*80)
        for name, _ in model.named_modules():
            print(name)
        print("="*80 + "\n")
        # 获取测试数据路径
        test_data_path = config['test_config'].get('test_data_path', '')
        if not test_data_path:
            test_data_path = Path(config['data_path']) / 'test'
        else:
            test_data_path = Path(test_data_path)
        
        print(f"测试数据路径: {test_data_path}")
        
        # 创建测试数据集
        test_dataset = TestMultiTaskDataset(
            test_data_path, 
            config=config, 
            split='test'
        )
        
        # 创建测试数据加载器
        test_batch_size = config['test_config']['save_features']['batch_size']
        test_loader = DataLoader(
            test_dataset, 
            batch_size=test_batch_size, 
            shuffle=False, 
            num_workers=config['num_workers']
        )
        
        print(f"测试数据集大小: {len(test_dataset)}")
        print(f"测试批次大小: {test_batch_size}")
        
        # 执行测试
        results = test_model(model, test_loader, device, config)
        
        # 评估结果
        metrics = evaluate_results(results)
        
        # 打印评估结果
        print("\n" + "="*60)
        print("测试结果")
        print("="*60)
        print(f"分类准确率: {metrics['classification_accuracy']:.2f}%")
        print(f"回归MSE: {metrics['regression_mse']:.6f}")
        print(f"回归MAE: {metrics['regression_mae']:.6f}")
        print(f"Beta2 - MSE: {metrics['mse_beta2']:.6f}, R²: {metrics['r2_beta2']:.6f}")
        print(f"Beta3 - MSE: {metrics['mse_beta3']:.6f}, R²: {metrics['r2_beta3']:.6f}")
        print("\n分类报告:")
        print(metrics['classification_report'])
        
        # 保存结果
        if config['test_config']['save_predictions']:
            # 保存预测结果
            predictions_file = test_output_dir / 'test_predictions.npz'
            np.savez(predictions_file, **results, **metrics)
            print(f"预测结果已保存到: {predictions_file}")
        
        # 绘制结果图
        if config['test_config']['save_plots']:
            plot_path = test_output_dir / 'test_results.png'
            plot_test_results(results, metrics, plot_path)
            print(f"结果图已保存到: {plot_path}")
        
        # 保存特征和数据
        save_features_and_data(results, test_output_dir, config)
        
        # 执行Grad-CAM分析
        if config['interpretability']['gradcam']['enabled']:
            perform_gradcam_analysis(model, test_loader, device, config, test_output_dir)
        
        print("\n" + "="*60)
        print("测试完成!")
        print("="*60)
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 恢复标准输出
        sys.stdout = custom_logger.terminal
        custom_logger.close()


if __name__ == '__main__':
    main()