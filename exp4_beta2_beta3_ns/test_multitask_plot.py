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
from train3 import (
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

plt.rcParams.update({
    "font.family": "serif",
    #"font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.linewidth": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
})
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

class PredictionDifferenceAnalyzer:
    """
    Prediction Difference Analyzer (PDA)
    参考: Zintgraf et al., Visualizing Deep Neural Network Decisions (ICLR 2017)
    """
    def __init__(self, model, task_type='classification', perturbation_size=8, perturbation_mode='zero', stride=4):
        """
        Args:
            model: 已训练模型
            task_type: 'classification' 或 'regression'
            perturbation_size: 遮挡区域的边长（像素）
            perturbation_mode: 遮挡方式 ('zero', 'mean', 'noise')
            stride: 遮挡滑动步长
        """
        self.model = model
        self.perturbation_size = perturbation_size
        self.perturbation_mode = perturbation_mode
        self.stride = stride

    def generate_pda(self, input_tensor,task_type='classification', target_class=None, regression_dim=None):
        """
        生成PDA重要性图

        Args:
            input_tensor: 输入图像 (1, C, H, W)
            target_class: 分类目标类别
            regression_dim: 回归任务的目标维度

        Returns:
            heatmap: 重要性热力图 (H, W)
        """
        self.model.eval()
        with torch.no_grad():
            reg_output, cls_output = self.model(input_tensor)

        if task_type == 'classification':
            if target_class is None:
                target_class = torch.argmax(cls_output, dim=1).item()
            baseline_pred = cls_output[0, target_class].item()
        else:
            if regression_dim is None:
                regression_dim = 0
            baseline_pred = reg_output[0, regression_dim].item()

        _, _, H, W = input_tensor.shape
        heatmap = np.zeros((H, W))

        # 遍历所有遮挡区域
        for y in range(0, H, self.stride):
            for x in range(0, W, self.stride):
                x_end = min(x + self.perturbation_size, W)
                y_end = min(y + self.perturbation_size, H)

                perturbed = input_tensor.clone()

                # 扰动该区域
                if self.perturbation_mode == 'zero':
                    perturbed[:, :, y:y_end, x:x_end] = 0
                elif self.perturbation_mode == 'mean':
                    mean_val = input_tensor.mean()
                    perturbed[:, :, y:y_end, x:x_end] = mean_val
                elif self.perturbation_mode == 'noise':
                    noise = torch.randn_like(perturbed[:, :, y:y_end, x:x_end]) * 0.1
                    perturbed[:, :, y:y_end, x:x_end] = noise

                # 再次前向传播
                with torch.no_grad():
                    reg_out, cls_out = self.model(perturbed)
                    if task_type == 'classification':
                        new_pred = cls_out[0, target_class].item()
                    else:
                        new_pred = reg_out[0, regression_dim].item()

                # 差值越大 → 该区域越重要
                diff = abs(baseline_pred - new_pred)
                heatmap[y:y_end, x:x_end] = diff

        # 归一化到 [0,1]
        heatmap -= heatmap.min()
        heatmap /= (heatmap.max() + 1e-8)
        return heatmap

class ConditionalPredictionDifferenceAnalyzer:
    """
    基于条件采样的 Prediction Difference Analyzer (PDA)
    参考: Zintgraf et al., Visualizing Deep Neural Network Decisions (ICLR 2017)
    
    核心思想:
    - 不是简单遮挡区域，而是用从条件分布中采样的值替换
    - 条件分布基于周围像素的统计信息
    - 更符合原论文的marginalization思想
    """
    
    def __init__(self, model, patch_size=8, stride=4, num_samples=10, 
                 sampling_mode='gaussian', context_size=3):
        """
        Args:
            model: 已训练模型
            patch_size: 被边缘化区域的大小
            stride: 滑动步长
            num_samples: 每个patch的采样次数(越多越准确但越慢)
            sampling_mode: 采样方式
                - 'gaussian': 基于周围像素的高斯分布
                - 'nearest': 从最近邻像素采样
                - 'inpainting': 使用简单的图像修复
            context_size: 用于估计条件分布的上下文窗口大小
        """
        self.model = model
        self.patch_size = patch_size
        self.stride = stride
        self.num_samples = num_samples
        self.sampling_mode = sampling_mode
        self.context_size = context_size
        
    def _get_context_stats(self, image, y, y_end, x, x_end):
        """
        获取patch周围区域的统计信息
        
        Args:
            image: 输入图像 (1, C, H, W)
            y, y_end, x, x_end: patch的坐标
            
        Returns:
            mean, std: 每个通道的均值和标准差
        """
        _, C, H, W = image.shape
        
        # 扩展的上下文区域
        ctx_y_start = max(0, y - self.context_size)
        ctx_y_end = min(H, y_end + self.context_size)
        ctx_x_start = max(0, x - self.context_size)
        ctx_x_end = min(W, x_end + self.context_size)
        
        # 提取上下文区域
        context = image[:, :, ctx_y_start:ctx_y_end, ctx_x_start:ctx_x_end].clone()
        
        # 将patch区域设为nan以便排除
        patch_y_in_ctx = y - ctx_y_start
        patch_y_end_in_ctx = y_end - ctx_y_start
        patch_x_in_ctx = x - ctx_x_start
        patch_x_end_in_ctx = x_end - ctx_x_start
        
        mask = torch.ones_like(context, dtype=torch.bool)
        mask[:, :, patch_y_in_ctx:patch_y_end_in_ctx, 
             patch_x_in_ctx:patch_x_end_in_ctx] = False
        
        # 计算周围像素的统计信息
        means = []
        stds = []
        for c in range(C):
            valid_pixels = context[0, c][mask[0, c]]
            if len(valid_pixels) > 0:
                means.append(valid_pixels.mean().item())
                stds.append(valid_pixels.std().item() + 1e-6)
            else:
                means.append(0.0)
                stds.append(1.0)
                
        return torch.tensor(means), torch.tensor(stds)
    
    def _sample_patch_gaussian(self, image, y, y_end, x, x_end):
        """从高斯分布采样替换patch"""
        _, C, _, _ = image.shape
        mean, std = self._get_context_stats(image, y, y_end, x, x_end)
        
        sampled = image.clone()
        for c in range(C):
            noise = torch.randn(y_end - y, x_end - x, device=image.device)
            sampled[0, c, y:y_end, x:x_end] = mean[c] + std[c] * noise
            
        return sampled
    
    def _sample_patch_nearest(self, image, y, y_end, x, x_end):
        """从最近邻像素随机采样"""
        _, C, H, W = image.shape
        
        # 获取边界像素
        boundary_pixels = []
        
        # 上边界
        if y > 0:
            boundary_pixels.append(image[:, :, y-1:y, x:x_end])
        # 下边界
        if y_end < H:
            boundary_pixels.append(image[:, :, y_end:y_end+1, x:x_end])
        # 左边界
        if x > 0:
            boundary_pixels.append(image[:, :, y:y_end, x-1:x])
        # 右边界
        if x_end < W:
            boundary_pixels.append(image[:, :, y:y_end, x_end:x_end+1])
            
        if not boundary_pixels:
            # 如果没有边界(整张图都是patch)，返回均值
            return image.clone()
        
        # 合并所有边界像素
        boundary = torch.cat([p.reshape(1, C, -1) for p in boundary_pixels], dim=2)
        
        sampled = image.clone()
        patch_h, patch_w = y_end - y, x_end - x
        
        for c in range(C):
            # 从边界像素中随机采样
            indices = torch.randint(0, boundary.shape[2], 
                                   (patch_h, patch_w), 
                                   device=image.device)
            sampled[0, c, y:y_end, x:x_end] = boundary[0, c, indices]
            
        return sampled
    
    def _sample_patch_inpainting(self, image, y, y_end, x, x_end):
        """使用简单的双线性插值修复"""
        _, C, H, W = image.shape
        sampled = image.clone()
        
        # 获取四个角的值进行双线性插值
        for c in range(C):
            # 边界值
            top_left = image[0, c, max(0, y-1), max(0, x-1)]
            top_right = image[0, c, max(0, y-1), min(W-1, x_end)]
            bottom_left = image[0, c, min(H-1, y_end), max(0, x-1)]
            bottom_right = image[0, c, min(H-1, y_end), min(W-1, x_end)]
            
            # 创建插值网格
            patch_h, patch_w = y_end - y, x_end - x
            y_interp = torch.linspace(0, 1, patch_h, device=image.device).view(-1, 1)
            x_interp = torch.linspace(0, 1, patch_w, device=image.device).view(1, -1)
            
            # 双线性插值
            interpolated = (
                top_left * (1 - y_interp) * (1 - x_interp) +
                top_right * (1 - y_interp) * x_interp +
                bottom_left * y_interp * (1 - x_interp) +
                bottom_right * y_interp * x_interp
            )
            
            # 添加一些随机扰动
            noise = torch.randn_like(interpolated) * 0.05
            sampled[0, c, y:y_end, x:x_end] = interpolated + noise
            
        return sampled
    
    def _sample_patch(self, image, y, y_end, x, x_end):
        """根据采样模式生成条件采样"""
        if self.sampling_mode == 'gaussian':
            return self._sample_patch_gaussian(image, y, y_end, x, x_end)
        elif self.sampling_mode == 'nearest':
            return self._sample_patch_nearest(image, y, y_end, x, x_end)
        elif self.sampling_mode == 'inpainting':
            return self._sample_patch_inpainting(image, y, y_end, x, x_end)
        else:
            raise ValueError(f"Unknown sampling mode: {self.sampling_mode}")
    
    def generate_pda(self, input_tensor, task_type='classification', 
                    target_class=None, regression_dim=None, batch_size=32):
        """
        生成基于条件采样的PDA重要性图
        
        Args:
            input_tensor: 输入图像 (1, C, H, W)
            task_type: 'classification' 或 'regression'
            target_class: 目标类别(分类任务)
            regression_dim: 目标维度(回归任务)
            batch_size: 批处理大小
            
        Returns:
            heatmap: 重要性热力图 (H, W)
        """
        self.model.eval()
        device = input_tensor.device
        
        # 输入验证
        if self.stride <= 0:
            raise ValueError("stride must be positive")
        
        _, C, H, W = input_tensor.shape
        
        # 1. 获取基线预测
        with torch.no_grad():
            reg_output, cls_output = self.model(input_tensor)
            
        if task_type == 'classification':
            target_class = target_class or torch.argmax(cls_output, dim=1).item()
            baseline_pred = cls_output[0, target_class].item()
        else:
            regression_dim = regression_dim or 0
            baseline_pred = reg_output[0, regression_dim].item()
        
        # 2. 初始化热力图
        importance_sum = np.zeros((H, W), dtype=np.float32)
        count_map = np.zeros((H, W), dtype=np.float32)
        
        # 3. 遍历所有patches
        all_samples = []
        all_coords = []
        
        for y in range(0, H, self.stride):
            for x in range(0, W, self.stride):
                x_end = min(x + self.patch_size, W)
                y_end = min(y + self.patch_size, H)
                
                # 对每个patch进行多次采样
                patch_diffs = []
                
                for _ in range(self.num_samples):
                    sampled_image = self._sample_patch(input_tensor, y, y_end, x, x_end)
                    all_samples.append(sampled_image)
                    all_coords.append((y, y_end, x, x_end))
        
        # 4. 批处理推理所有采样
        all_predictions = []
        
        with torch.no_grad():
            for i in range(0, len(all_samples), batch_size):
                batch = torch.cat(all_samples[i:i+batch_size], dim=0).to(device)
                reg_out, cls_out = self.model(batch)
                
                if task_type == 'classification':
                    preds = cls_out[:, target_class].cpu().numpy()
                else:
                    preds = reg_out[:, regression_dim].cpu().numpy()
                    
                all_predictions.extend(preds)
        
        # 5. 计算每个patch的平均重要性
        idx = 0
        for y in range(0, H, self.stride):
            for x in range(0, W, self.stride):
                x_end = min(x + self.patch_size, W)
                y_end = min(y + self.patch_size, H)
                
                # 计算该patch所有采样的平均预测差异
                patch_preds = all_predictions[idx:idx+self.num_samples]
                avg_diff = np.mean([abs(baseline_pred - pred) for pred in patch_preds])
                
                importance_sum[y:y_end, x:x_end] += avg_diff
                count_map[y:y_end, x:x_end] += 1
                
                idx += self.num_samples
        
        # 6. 归一化(处理重叠区域)
        heatmap = importance_sum / (count_map + 1e-8)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap



def perform_conditional_pda_analysis(model, test_loader, device, config, output_dir):
    """执行基于条件采样的PDA分析"""
    pda_config = config['interpretability']['pda']
    if not pda_config['enabled']:
        return
    
    print("开始条件采样 Prediction Difference Analyzer 分析...")
    
    pda_dir = output_dir / pda_config['output_dir']
    pda_dir.mkdir(parents=True, exist_ok=True)
    
    if pda_config['task_specific']['regression']:
        (pda_dir / 'regression').mkdir(exist_ok=True)
    if pda_config['task_specific']['classification']:
        (pda_dir / 'classification').mkdir(exist_ok=True)
    
    # 创建分析器
    analyzer = ConditionalPredictionDifferenceAnalyzer(
        model,
        patch_size=pda_config.get('patch_size', 8),
        stride=pda_config.get('stride', 4),
        num_samples=pda_config.get('num_conditional_samples', 10),  # 每个patch的采样次数
        sampling_mode=pda_config.get('sampling_mode', 'gaussian'),  # 采样模式
        context_size=pda_config.get('context_size', 3)
    )
    
    inference_batch_size = pda_config.get('internal_batch_size', 32)
    num_samples = pda_config.get('num_samples', -1)
    samples_processed = 0
    
    pda_loader = tqdm(test_loader, desc="Conditional PDA Analysis")
    
    for batch_idx, (inputs, reg_labels, cls_labels, file_paths, processed_images) in enumerate(pda_loader):
        if num_samples != -1 and samples_processed >= num_samples:
            break
        
        for i in range(inputs.size(0)):
            if num_samples != -1 and samples_processed >= num_samples:
                break
            
            file_stem = Path(file_paths[i]).stem
            single_input = inputs[i:i+1].to(device)
            processed_image = processed_images[i]
            
            # 分类任务
            if pda_config['task_specific']['classification']:
                heatmap = analyzer.generate_pda(
                    single_input,
                    task_type='classification',
                    batch_size=inference_batch_size
                )
                save_pda_visualization(
                    heatmap, processed_image,
                    pda_dir / 'classification' / f'{file_stem}_cls_conditional_pda.png',
                    pda_config
                )
            
            # 回归任务
            if pda_config['task_specific']['regression']:
                for dim, name in enumerate(['beta2', 'beta3']):
                    heatmap = analyzer.generate_pda(
                        single_input,
                        task_type='regression',
                        regression_dim=dim,
                        batch_size=inference_batch_size
                    )
                    save_pda_visualization(
                        heatmap, processed_image,
                        pda_dir / 'regression' / f'{file_stem}_{name}_conditional_pda.png',
                        pda_config
                    )
            
            samples_processed += 1
            pda_loader.set_postfix({'processed': samples_processed})
    
    print(f"条件采样 PDA 分析完成，共处理 {samples_processed} 个样本。")


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


import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

def plot_test_results(results, metrics, output_path, stats=None):
    """
    绘制测试结果。
    
    Args:
        results: 包含预测值和真实值的字典 (numpy array)
        metrics: 指标字典
        output_path: 保存路径
        stats: (新增) 数据集的统计量字典 {'mean': tensor, 'std': tensor}，用于反归一化
    """
    base_path = Path(output_path)
    parent_dir = base_path.parent
    stem = base_path.stem 
    
    # 获取原始数据 (此时是归一化后的状态)
    reg_preds = results['reg_predictions'].copy() # copy防止修改原数据
    reg_labels = results['reg_labels'].copy()
    
    # --- 新增：反向归一化逻辑 ---
    if stats is not None:
        # 1. 将 tensor 转换为 numpy，并确保在 CPU 上
        mean = stats['mean'].cpu().numpy()
        std = stats['std'].cpu().numpy()
        
        # 2. 应用公式: Original = Normalized * Std + Mean
        reg_preds = reg_preds * std + mean
        reg_labels = reg_labels * std + mean
        print("✅ 已执行反向归一化：预测值和真实值已恢复为物理数值。")
    else:
        print("⚠️ 警告：未传入 stats，绘图将使用归一化后的数值。")

    # ---------------------------------------------------------
    # 1. 绘制 Beta2 回归图 (使用 Hist2d 密度图)
    # ---------------------------------------------------------
    fig_b2, ax_b2 = plt.subplots(figsize=(6, 5)) # 稍微增加宽度以容纳colorbar
    
    # [修改处] 使用 hist2d 替代 scatter
    # bins=50: 网格密度，可根据需要调整
    # cmap='Blues': 对应学姐图片的蓝色风格
    # cmin=1: 计数小于1的格子不显示颜色（显示为白色背景）
    h_b2 = ax_b2.hist2d(reg_labels[:, 0], reg_preds[:, 0], bins=10, cmap='Blues', cmin=1)
    
    # [修改处] 添加 Colorbar
    cb_b2 = fig_b2.colorbar(h_b2[3], ax=ax_b2)
    cb_b2.set_label('Counts')

    # 计算坐标范围（保持你原来的逻辑）
    min_b2 = min(reg_labels[:, 0].min(), reg_preds[:, 0].min())
    max_b2 = max(reg_labels[:, 0].max(), reg_preds[:, 0].max())
    margin_b2 = (max_b2 - min_b2) * 0.05
    
    # 绘制对角线
    ax_b2.plot([min_b2-margin_b2, max_b2+margin_b2], 
               [min_b2-margin_b2, max_b2+margin_b2], 
               'r--', lw=2.5)
    
    ax_b2.set_xlabel(r'True $\beta_2$')
    ax_b2.set_ylabel(r'Predicted $\beta_2$')
    ax_b2.set_xlim(min_b2-margin_b2, max_b2+margin_b2)
    ax_b2.set_ylim(min_b2-margin_b2, max_b2+margin_b2)
    
    stats_text_b2 = (f"$R^2 = {metrics['r2_beta2']:.4f}$")
    ax_b2.text(0.05, 0.95, stats_text_b2, transform=ax_b2.transAxes, va='top', fontsize=14,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # 图例 (只保留对角线的图例，因为散点已变成密度图)
    ax_b2.legend(frameon=False, loc='lower right')
    
    save_name_b2 = parent_dir / f"{stem}_Beta2.pdf"
    # 注意：使用tight_layout会自动调整colorbar的位置，防止重叠
    plt.figure(fig_b2.number)
    plt.tight_layout()
    plt.savefig(save_name_b2, dpi=300, bbox_inches='tight')
    plt.close(fig_b2)
    print(f"✅ Saved: {save_name_b2}")

    # ---------------------------------------------------------
    # 2. 绘制 Beta3 回归图 (使用 Hist2d 密度图)
    # ---------------------------------------------------------
    fig_b3, ax_b3 = plt.subplots(figsize=(6, 5))
    
    # [修改处] 使用 hist2d 替代 scatter
    h_b3 = ax_b3.hist2d(reg_labels[:, 1], reg_preds[:, 1], bins=10, cmap='Blues', cmin=1)
    
    # [修改处] 添加 Colorbar
    cb_b3 = fig_b3.colorbar(h_b3[3], ax=ax_b3)
    cb_b3.set_label('Counts')
    
    min_b3 = min(reg_labels[:, 1].min(), reg_preds[:, 1].min())
    max_b3 = max(reg_labels[:, 1].max(), reg_preds[:, 1].max())
    margin_b3 = (max_b3 - min_b3) * 0.05
    
    ax_b3.plot([min_b3-margin_b3, max_b3+margin_b3], 
               [min_b3-margin_b3, max_b3+margin_b3], 
               'r--', lw=2.5)
    
    ax_b3.set_xlabel(r'True $\beta_3$')
    ax_b3.set_ylabel(r'Predicted $\beta_3$')
    ax_b3.set_xlim(min_b3-margin_b3, max_b3+margin_b3)
    ax_b3.set_ylim(min_b3-margin_b3, max_b3+margin_b3)
    
    stats_text_b3 = (f"$R^2 = {metrics['r2_beta3']:.4f}$")
    ax_b3.text(0.05, 0.95, stats_text_b3, transform=ax_b3.transAxes, va='top', fontsize=14,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    ax_b3.legend(frameon=False, loc='lower right')
    
    save_name_b3 = parent_dir / f"{stem}_Beta3.pdf"
    save_name_b3_png = parent_dir / f"{stem}_Beta3.png"
    plt.figure(fig_b3.number)
    plt.tight_layout()
    plt.savefig(save_name_b3, dpi=300, bbox_inches='tight')
    plt.savefig(save_name_b3_png, dpi=300, bbox_inches='tight')
    plt.close(fig_b3)
    print(f"✅ Saved: {save_name_b3}")

    # ---------------------------------------------------------
    # 3. 绘制混淆矩阵 (保持不变)
    # ---------------------------------------------------------
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    
    cm = metrics['confusion_matrix']
    class_names = ['Halo', 'Skin', 'None'] 
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, 
                ax=ax_cm, cbar=False, annot_kws={"size": 16})
    
    ax_cm.set_xlabel('Predicted Label')
    ax_cm.set_ylabel('True Label')
    
    save_name_cm = parent_dir / f"{stem}_NeutronSkin_CM.pdf"
    plt.figure(fig_cm.number)
    plt.tight_layout()
    plt.savefig(save_name_cm, dpi=300, bbox_inches='tight')
    plt.close(fig_cm)
    print(f"✅ Saved: {save_name_cm}")
def _save_overlay_only(heatmap, processed_image, output_path, config):
    """
    保存带有坐标轴、Colorbar 和出版级样式的 Overlay 图片。
    风格仿照 save_paper_plot，坐标范围 -0.17 到 0.17。
    """
    # ---------------------------------------------------------
    # 1. 数据准备 (Image 反标准化)
    # ---------------------------------------------------------
    if isinstance(processed_image, torch.Tensor):
        img_tensor = processed_image.cpu()
    else:
        img_tensor = torch.from_numpy(processed_image)
        
    # [C, H, W] -> [H, W, C]
    img_display = img_tensor.numpy().transpose(2, 1, 0)
    
    # ImageNet 标准化参数
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_display = std * img_display + mean
    img_display = np.clip(img_display, 0, 1)
    
    height, width = img_display.shape[:2]

    # ---------------------------------------------------------
    # 2. 热力图处理
    # ---------------------------------------------------------
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.cpu().numpy()
        
    # Resize 到图片大小
    heatmap_resized = cv2.resize(heatmap, (width, height))
    
    # 保持你原有的转置逻辑 (如果需要)
    heatmap_resized = np.transpose(heatmap_resized, (1, 0))
    
    # 归一化热力图 (0-1) 用于显示
    # 注意：这里我们归一化是为了 color mapping，
    # 但如果 heatmap 本身有物理意义的数值范围，可以不归一化，直接调整 vmin/vmax
    h_min, h_max = heatmap_resized.min(), heatmap_resized.max()
    if h_max - h_min > 0:
        heatmap_norm = (heatmap_resized - h_min) / (h_max - h_min)
    else:
        heatmap_norm = heatmap_resized

    # ---------------------------------------------------------
    # 3. 坐标系与方向处理 (关键)
    # ---------------------------------------------------------
    # 目标风格使用了 origin='lower' (y轴向上)。
    # 计算机视觉图像通常 (0,0) 在左上角。
    # 为了在 'lower' 模式下显示正常，我们需要上下翻转数据。
    img_for_plot = np.flipud(img_display)
    heatmap_for_plot = np.flipud(heatmap_norm)
    
    # 坐标范围
    p_limit = 0.17
    extent = [-p_limit, p_limit, -p_limit, p_limit]

    # ---------------------------------------------------------
    # 4. 绘图 (仿照 save_paper_plot 风格)
    # ---------------------------------------------------------
    # 自动生成标题
    stem = Path(output_path).stem
    if 'beta2' in stem.lower():
        title_text = r'(a) $\beta_2$ Overlay'
    elif 'beta3' in stem.lower():
        title_text = r'(b) $\beta_3$ Overlay'
    elif 'cls' in stem.lower():
        title_text = r'(c) Class Overlay'
    else:
        title_text = "Overlay Analysis"

    fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')

    # A. 绘制底层背景图 (灰度或彩色)
    # 使用 gray cmap 如果图像是单通道，否则直接显示 RGB
    ax.imshow(img_for_plot, extent=extent, origin='lower')

    # B. 绘制顶层热力图 (带透明度)
    alpha = config.get('alpha', 0.4) # 获取透明度配置
    colormap = config.get('colormap', 'jet')
    
    # 绘制热力图层
    im = ax.imshow(heatmap_for_plot, extent=extent, origin='lower',
                   cmap=colormap, alpha=alpha, vmin=0, vmax=1)
    
    # C. 添加 Colorbar
    cbar = fig.colorbar(im, ax=ax)
    
    # ---------------------------------------------------------
    # 5. 样式精修 (完全复制参考代码)
    # ---------------------------------------------------------
    # Zoom / Limits
    ax.set_xlim(-p_limit, p_limit)
    ax.set_ylim(-p_limit, p_limit)
    
    # 标题居中
    ax.set_title(title_text, color='black', weight='bold', fontsize=18, loc='center')
    
    # 轴标签
    ax.set_xlabel(r'$p_x$ (GeV/c)', color='black', fontsize=14)
    ax.set_ylabel(r'$p_y$ (GeV/c)', color='black', fontsize=14)

    # 刻度与边框
    ax.tick_params(direction='in', colors='black', which='both', top=True, right=True, labelsize=12)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)
    
    # Colorbar 样式
    cbar.ax.yaxis.set_tick_params(color='black')
    cbar.outline.set_edgecolor('black')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black', fontsize=12)

    plt.tight_layout()
    
    # 保存
    plt.savefig(output_path, dpi=300, facecolor='white', edgecolor='none')
    plt.close(fig)

def perform_pda_analysis(model, test_loader, device, config, output_dir):
    """执行 Prediction Difference Analysis (PDA)"""
    pda_config = config['interpretability']['pda']
    if not pda_config['enabled']:
        return

    print("开始 Prediction Difference Analyzer 分析...")

    pda_dir = output_dir / pda_config['output_dir']
    pda_dir.mkdir(parents=True, exist_ok=True)

    if pda_config['task_specific']['regression']:
        (pda_dir / 'regression').mkdir(exist_ok=True)
    if pda_config['task_specific']['classification']:
        (pda_dir / 'classification').mkdir(exist_ok=True)

    analyzer = PredictionDifferenceAnalyzer(
        model,
        perturbation_size=pda_config.get('patch_size', 8),
        stride=pda_config.get('stride', 4),
        perturbation_mode=pda_config.get('perturbation_mode', 'mean')
    )

    num_samples = pda_config['num_samples']
    samples_processed = 0
    pda_loader = tqdm(test_loader) 
    for batch_idx, (inputs, reg_labels, cls_labels, file_paths, processed_images) in enumerate(pda_loader):
        if num_samples != -1 and samples_processed >= num_samples:
            break
        inputs = inputs.to(device)
        for i in range(inputs.size(0)):
            if num_samples != -1 and samples_processed >= num_samples:
                break

            file_stem = Path(file_paths[i]).stem
            single_input = inputs[i:i+1]
            processed_image = processed_images[i]

            if pda_config['task_specific']['classification']:
                heatmap = analyzer.generate_pda(single_input, task_type='classification')
                save_pda_visualization(
                    heatmap, processed_image,
                    pda_dir / 'classification' / f'{file_stem}_cls_pda.png',
                    pda_config
                )

            if pda_config['task_specific']['regression']:
                for dim, name in enumerate(['beta2', 'beta3']):
                    heatmap = analyzer.generate_pda(single_input, task_type='regression', regression_dim=dim)
                    save_pda_visualization(
                        heatmap, processed_image,
                        pda_dir / 'regression' / f'{file_stem}_{name}_pda.png',
                        pda_config
                    )

            samples_processed += 1
            #if samples_processed % 10 == 0:
            #    print(f"PDA 进度: {samples_processed}/{num_samples if num_samples != -1 else '?'}")

    print(f"PDA 分析完成，共处理 {samples_processed} 个样本。")
def save_pda_visualization(heatmap, processed_image, output_path, config):
    """保存PDA结果可视化 (仅 Overlay)"""
    _save_overlay_only(heatmap, processed_image, output_path, config)
    print(f"✅ Saved PDA Overlay: {output_path}")

def save_gradcam_visualization(cam, processed_image, output_path, config):
    """保存Grad-CAM可视化结果 (仅 Overlay)"""
    # Grad-CAM 传入的 cam 通常尺寸较小，内部辅助函数会处理 resize
    _save_overlay_only(cam, processed_image, output_path, config)
    print(f"✅ Saved Grad-CAM Overlay: {output_path}")

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
    with open('config_test_inch_plot.yaml', 'r', encoding='utf-8') as f:
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
            if hasattr(test_dataset, "stats"):
                plot_test_results(results, metrics, plot_path,
                                stats=test_dataset.stats)
            else:
                plot_test_results(results, metrics, plot_path)
            print(f"结果图已保存到: {plot_path}")
        
        # 保存特征和数据
        save_features_and_data(results, test_output_dir, config)
        
        # 执行Grad-CAM分析
        if config['interpretability']['gradcam']['enabled']:
            perform_gradcam_analysis(model, test_loader, device, config, test_output_dir)
        if config['interpretability']['pda']['enabled']:
            perform_conditional_pda_analysis(model, test_loader, device, config, test_output_dir)
        
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