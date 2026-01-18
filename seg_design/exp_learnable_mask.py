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
import math




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
        # 情况 1: 原有的 MultiTaskResNetMLP 模型 (基于 ResNet)
        if hasattr(self.model, 'feature_extractor'):
            for name, module in self.model.feature_extractor.feature_proj.named_modules():
                if isinstance(module, nn.Linear):
                    last_linear = module
            return last_linear.weight
            
        # 情况 2: 带有 Router 的 PhysicalRoutingModel
        elif hasattr(self.model, 'router'):
            # Router 的 logits 是所有任务共享的瓶颈
            return self.model.router.logits
            
        # [新增] 情况 3: 全图验证模型 PhysicalSanityCheckModel
        elif hasattr(self.model, 'adapter'):
            # 我们需要找到 adapter 里的可学习参数
            
            # A. 如果是 LearnablePhysicsAdapter (AI 物理学家)
            if hasattr(self.model.adapter, 'operator_generator'):
                # 返回生成网络的最后一层权重
                # operator_generator 是一个 Sequential
                return self.model.adapter.operator_generator[-1].weight
                
            # B. 如果是 BasisPhysicsAdapter (基函数版)
            elif hasattr(self.model.adapter, 'coeffs'):
                return self.model.adapter.coeffs
                
            # C. 如果是 MomentAdapter/CoarseGrainedAdapter (静态算子)
            else:
                # 警告：静态算子没有共享参数，GradNorm 在原理上是失效的！
                # 因为前向传播没有参数 w_shared，所以 dL/dw_shared 永远是 0 或不存在。
                # 为了防止代码崩溃，我们返回第一层 Task Head 的权重作为替补，
                # 但建议这种情况下直接关闭 GradNorm (设 alpha=0 或改用简单加权)。
                print("[Warning] GradNorm 检测到静态 Adapter，无共享参数。GradNorm 机制将失效。")
                if hasattr(self.model, 'head_skin'):
                    return self.model.head_skin.net[1].weight # 返回 Head 的第一层作为占位
                else:
                    # 最后的保底，生成一个 dummy
                    return torch.zeros(1, requires_grad=True, device=next(self.model.parameters()).device)
            
        else:
            raise AttributeError("无法找到 GradNorm 所需的共享参数。"
                                 "检查模型是否包含 feature_extractor, router 或 adapter")
    
    def compute_grad_norm(self, losses):
        """
        计算并应用GradNorm算法
        (保持原有逻辑不变)
        """
        # 如果是第一次运行，初始化损失记录
        if self.initial_losses is None:
            self.initial_losses = losses.clone().detach()
            self.running_losses = losses.clone().detach()
        
        # 更新运行平均损失
        decay = 0.1
        self.running_losses = (1 - decay) * self.running_losses + decay * losses.detach()
        
        # 获取当前任务权重
        weights = self.task_weights()
        
        # 计算加权损失
        weighted_loss = torch.sum(weights * losses)
        
        # 清零梯度 (针对共享参数)
        if self.shared_params.grad is not None:
            self.shared_params.grad.zero_()
        
        # 分别计算每个任务对共享参数的梯度
        grad_norms = []
        for i, loss in enumerate(losses):
            # 计算当前任务对共享参数的梯度
            # retain_graph=True 是必须的，因为我们要对同一个图backward多次
            task_grad = torch.autograd.grad(
                weights[i] * loss, 
                self.shared_params, 
                retain_graph=True,
                create_graph=True,
                allow_unused=True # 允许某些任务不通过 Router (例如 Dummy Task)
            )[0]
            
            # 处理 allow_unused=True 的情况 (防止 None)
            if task_grad is None:
                grad_norm = torch.zeros(1, device=weights.device)
            else:
                grad_norm = torch.norm(task_grad)
                
            grad_norms.append(grad_norm)
        
        grad_norms = torch.stack(grad_norms)
        
        # 计算相对损失率
        loss_ratios = self.running_losses / self.initial_losses
        
        # 计算平均梯度范数
        mean_grad_norm = torch.mean(grad_norms)
        
        # 计算目标梯度范数
        relative_rates = torch.pow(loss_ratios, self.alpha)
        mean_relative_rate = torch.mean(relative_rates)
        target_grad_norms = mean_grad_norm * (relative_rates / mean_relative_rate)
        
        # 计算GradNorm损失
        gradnorm_loss = torch.sum(torch.abs(grad_norms - target_grad_norms))
        
        return weighted_loss, gradnorm_loss
    
    def get_current_weights(self):
        return self.task_weights().detach().cpu().numpy()




class ExperimentalFeasibilityLoss(nn.Module):
    def __init__(self, lambda_sep=0.1, lambda_size=1.0, min_dist=0.5, target_sigma=0.15):
        super().__init__()
        self.lambda_sep = lambda_sep
        self.lambda_size = lambda_size
        self.min_dist = min_dist
        self.target_sigma = target_sigma # [新增] 期望的最小尺寸 (~33像素)

    def forward(self, model):
        generator = model.region_generator
        locs = generator.locs
        # 限制 sigma 的最大值，防止它虽然不产生负loss，但在内部数值爆炸
        # 这里的 exp 出来最大是 1.0 (全图宽的一半)
        sigmas = torch.exp(torch.clamp(generator.log_sigma, max=0.0)) 
        
        # --- A. 排斥 Loss (不变) ---
        dists = torch.pdist(locs, p=2)
        repulsion_loss = torch.mean(1.0 / (dists + 1e-6))
        
        # --- B. [关键修改] 大小 Loss (Hinge Loss) ---
        # 旧代码: size_loss = -torch.mean(sigmas)  <-- 导致负无穷
        # 新代码: 只惩罚小于 target_sigma 的情况
        # 如果 sigma > 0.15，ReLU 输出 0，Loss 为 0。
        # 如果 sigma < 0.15，Loss 为正数，迫使它变大。
        size_loss = torch.mean(torch.relu(self.target_sigma - sigmas))
        
        # --- C. 边界 Loss (不变) ---
        boundary_loss = torch.mean(torch.relu(torch.abs(locs) - 0.9)**2)

        # 总 Loss (保证永远是非负数)
        total_reg_loss = (self.lambda_sep * repulsion_loss + 
                          self.lambda_size * size_loss + 
                          10.0 * boundary_loss)
                          
        return total_reg_loss, repulsion_loss, size_loss



# ===================================================================
# 1. 数据预处理模块
# ===================================================================

class GaussianSpatialNoise:
    """
    添加基于空间位置的高斯噪声 (实际上是一个高斯偏置场)
    公式: Noise = C * e^(-4 * r^2)
    其中:
      1. r 为像素距离中心的物理距离 = 像素距离 * pixel_scale
      2. C 为动态计算值 = 当前图片像素均值 * factor (默认0.01)
    """
    def __init__(self, factor=0.01, pixel_scale=1.0/645.0, enabled=True):
        self.factor = factor      # 均值的倍数，例如 0.01
        self.pixel_scale = pixel_scale
        self.enabled = enabled

    def __call__(self, tensor):
        if not self.enabled:
            return tensor

        # 获取当前张量形状 (C, H, W) 或 (H, W)
        if tensor.dim() == 3:
            h, w = tensor.shape[1], tensor.shape[2]
        else:
            h, w = tensor.shape[0], tensor.shape[1]

        # -------------------------------------------------------
        # 1. 动态计算 C = mean(tensor) * 0.01
        # -------------------------------------------------------
        # 计算整张图（所有通道）的平均值
        image_mean = tensor.mean()
        C = image_mean * self.factor

        # -------------------------------------------------------
        # 2. 计算网格坐标与距离 r
        # -------------------------------------------------------
        center_h = (h - 1) / 2.0
        center_w = (w - 1) / 2.0

        y = torch.arange(h, device=tensor.device, dtype=tensor.dtype) - center_h
        x = torch.arange(w, device=tensor.device, dtype=tensor.dtype) - center_w
        
        # 生成网格
        grid_y, grid_x = torch.meshgrid(y, x)

        # r^2 = (y^2 + x^2) * scale^2
        dist_sq_pixels = grid_y**2 + grid_x**2
        r_sq = dist_sq_pixels * (self.pixel_scale ** 2)

        # -------------------------------------------------------
        # 3. 计算高斯噪声并叠加
        # -------------------------------------------------------
        # Noise = C * e^(-4 * r^2)
        noise = C * torch.exp(-4 * r_sq)

        # noise 形状为 (H, W)，会自动广播到 tensor 的 (Channels, H, W)
        return tensor + noise

    def __repr__(self):
        return f"{self.__class__.__name__}(factor={self.factor}, scale={self.pixel_scale:.6f})"


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
        noise_cfg = self.config.get('gaussian_spatial_noise', {})
        if noise_cfg.get('enabled', False):
            # 这里不再读取 C，而是读取 factor (默认0.01)
            factor = noise_cfg.get('factor', 0.01)
            scale = noise_cfg.get('scale', 1.0/645.0)
            
            transforms_list.append(GaussianSpatialNoise(factor=factor, pixel_scale=scale, enabled=True))
            print(f"启用高斯空间噪声: C=mean*{factor}, scale={scale:.6f}")

        # 2. 对数归一化 (如果启用)
        if self.config.get('log_normalization', {}).get('enabled', False):
            epsilon = self.config['log_normalization'].get('epsilon', 1e-12)
            transforms_list.append(LogNormalization(epsilon=epsilon, enabled=True))
            print(f"启用对数归一化: epsilon={epsilon}")
        else:
            transforms_list.append(torch.log1p)
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
# 2. 数据集类（修改为多任务）
# ===================================================================

class MultiTaskInterferenceDataset(Dataset):
    def __init__(self, data_path, config, split='train', stats=None):
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
        self.regression_labels, self.classification_labels = self._parse_labels()
        
        # --- 新增：统计量计算逻辑 ---
        if stats is not None:
            # 如果传入了统计量（通常是验证集使用训练集的统计量），直接使用
            self.stats = stats
            print(f"'{self.split}' 使用传入的统计量进行归一化。")
        else:
            # 否则自动计算（训练集）
            # 将所有标签堆叠计算 mean 和 std
            all_reg_labels = torch.stack(self.regression_labels)
            self.stats = {
                'mean': torch.mean(all_reg_labels, dim=0),
                'std': torch.std(all_reg_labels, dim=0) + 1e-8 # 加一个小数值防止除以0
            }
            print(f"'{self.split}' 计算统计量完成:")
            print(f"  - Beta2 Mean: {self.stats['mean'][0]:.4f}, Std: {self.stats['std'][0]:.4f}")
            print(f"  - Beta3 Mean: {self.stats['mean'][1]:.4f}, Std: {self.stats['std'][1]:.4f}")

    def _parse_labels(self):
        # ... (这部分代码保持不变) ...
        # 复制你原来的 _parse_labels 代码即可
        beta_pattern = re.compile(r"beta2_([\d.]+)_beta3_([\d.]+)")
        class_pattern = re.compile(r"class(\d)")
        
        beta_pairs = []
        class_labels = []
        
        for file_path in self.file_paths:
            beta_match = beta_pattern.search(file_path.stem)
            if beta_match:
                beta2 = float(beta_match.group(1))
                beta3 = float(beta_match.group(2))
                beta_pairs.append((beta2, beta3))
            else:
                continue
            
            class_match = class_pattern.search(file_path.stem)
            if class_match:
                class_id = int(class_match.group(1))
                class_labels.append(class_id)
            else:
                beta_pairs.pop()
                continue

        regression_labels = [torch.tensor([p[0], p[1]], dtype=torch.float32) for p in beta_pairs]
        return regression_labels, class_labels

    def __len__(self):
        return len(self.regression_labels)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image_data = np.load(file_path)
        image_tensor = torch.from_numpy(image_data.astype(np.float32)).unsqueeze(0)
        image_tensor = self.preprocessor(image_tensor)
        
        # 获取原始标签
        raw_reg_label = self.regression_labels[idx]
        classification_label = self.classification_labels[idx]
        
        # --- 新增：应用标准化 (Z-Score Normalization) ---
        # 公式: (x - mean) / std
        normalized_reg_label = (raw_reg_label - self.stats['mean']) / self.stats['std']
        
        return image_tensor, normalized_reg_label, classification_label


# ===================================================================
# 3. 多任务模型定义
# ===================================================================

# ==========================================
# 模块 A: gaussian式分割器 (GaussianMask)
# ==========================================
class LearnableMultiRegionGenerator(nn.Module):
    """
    [Module A - Spatial Discovery] 多区域探针生成器
    
    生成 num_regions 个独立的高斯 Mask。
    每个 Mask 代表一个独立的“探测窗口”。
    AI 将学习将这些窗口移动到动量谱的关键位置（如：一个去亮斑中心，一个去暗纹处）。
    """
    def __init__(self, H=224, W=224, num_regions=8):
        super().__init__()
        self.num_regions = num_regions
        self.H, self.W = H, W
        
        # --- 初始化参数 ---
        # 1. 位置 (mu_x, mu_y): 
        # 为了防止一开始所有探针叠在一起，我们用均匀网格初始化它们
        # 范围 [-0.8, 0.8] 避免太靠边
        initial_locs = torch.rand(num_regions, 2) * 1.6 - 0.8
        self.locs = nn.Parameter(initial_locs)
        
        # 2. 宽度 (sigma): 
        # 初始化为较小的区域，让它专注于局部细节
        # log(-2.0) approx 0.13 (相对于 [-1,1] 坐标系)
        self.log_sigma = nn.Parameter(torch.ones(num_regions) * -2.0)
        
        # 3. 形状参数 (可选): 允许变成椭圆 (sigma_x != sigma_y) 增加灵活性
        # 这里先保持圆形，简单一点

        # --- 坐标网格 Buffer ---
        y = torch.linspace(-1, 1, H)
        x = torch.linspace(-1, 1, W)
        gy, gx = torch.meshgrid(y, x)
        self.register_buffer('grid_y', gy.unsqueeze(0).unsqueeze(0)) # (1, 1, H, W)
        self.register_buffer('grid_x', gx.unsqueeze(0).unsqueeze(0)) # (1, 1, H, W)

    def forward(self, batch_size):
        """
        Returns: (B, num_regions, H, W)
        """
        # (num_regions, 1, 1)
        mu_x = self.locs[:, 1].view(self.num_regions, 1, 1)
        mu_y = self.locs[:, 0].view(self.num_regions, 1, 1)
        if self.training:
            # 抖动幅度 0.02 (约等于 224*0.02 = 4.5 个像素)
            jitter = torch.randn_like(mu_x) * 0.02 
            mu_x = mu_x + jitter
            mu_y = mu_y + jitter
        sigma = torch.exp(torch.clamp(self.log_sigma, min=-1)).view(self.num_regions, 1, 1)
        
        # 计算高斯分布
        # Broadcasting: Grid(1,1,H,W) - Mu(N,1,1) -> (1,N,H,W)
        dist_sq = (self.grid_x - mu_x)**2 + (self.grid_y - mu_y)**2
        
        # 生成 Masks: (1, num_regions, H, W)
        masks = torch.exp(-dist_sq / (2 * sigma**2 + 1e-6))
        
        # 扩展到 Batch: (B, num_regions, H, W)
        batch_masks = masks.expand(batch_size, -1, -1, -1)
        
        return batch_masks



# ==========================================
# 模块 B: Gumbel-Sinkhorn 路由器
# ==========================================
class GumbelSinkhornRouter(nn.Module):
    def __init__(self, num_masks=3, num_tasks=3, n_iters=3, init_temp=1.0):
        super().__init__()
        self.num_masks = num_masks
        self.num_tasks = num_tasks
        self.n_iters = n_iters
        
        # Logits: (Task, Mask)
        # 初始化为微小随机数
        self.logits = nn.Parameter(torch.randn(num_tasks, num_masks) * 0.1)
        
        # Temperature buffer
        self.register_buffer('temperature', torch.tensor(init_temp))

    def set_temperature(self, temp):
        self.temperature.fill_(temp)

    def _sample_gumbel(self, shape):
        U = torch.rand(shape, device=self.logits.device)
        return -torch.log(-torch.log(U + 1e-20) + 1e-20)

    def _sinkhorn(self, log_alpha):
        for _ in range(self.n_iters):
            # Row Norm (Task constraint)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)
            # Col Norm (Mask constraint)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-2, keepdim=True)
        return torch.exp(log_alpha)

    def forward(self, masks, hard=False):
        """
        masks: (B, 3, H, W)
        Returns:
            routed_masks: (B, 3, H, W)
            P: (3, 3) Assignment Matrix
        """
        # 1. Logits -> Soft P
        if self.training:
            noise = self._sample_gumbel(self.logits.shape)
            noisy_logits = (self.logits + noise) / self.temperature
        else:
            noisy_logits = self.logits / self.temperature
            
        P = self._sinkhorn(noisy_logits) # (Tasks, Masks)

        # 2. Hard Routing (Straight-Through Estimator)
        if hard:
            P_hard = torch.zeros_like(P)
            P_hard.scatter_(1, P.argmax(dim=1, keepdim=True), 1.0)
            P = (P_hard - P).detach() + P
        
        # 3. Apply Routing
        # Einstein Summation: Task_t = Sum(P_tm * Mask_m)
        routed_masks = torch.einsum('tm, bmhw -> bthw', P, masks)
        
        return routed_masks, P

# ==========================================
# 模块 C: 物理算子适配器 (Physics OpLib)
# ==========================================
class LearnablePhysicsOperatorAdapter(nn.Module):
    """
    [V4.0 Learnable] 可学习物理算子适配器
    
    核心机制:
    1. 预计算物理基底 (Basis Bank): 包含 pT 的幂次项和 phi 的傅里叶项。
    2. 算子生成 (Operator Generation): 通过 1x1 Conv (Linear) 学习基底的线性组合。
    3. 全局积分 (Global Integration): 模拟物理实验中的积分测量，极度抗噪。
    """
    def __init__(self, num_learnable_features=16, max_harmonics=4, max_power=4):
        super().__init__()
        self.num_features = num_learnable_features-1
        self.max_harmonics = max_harmonics
        self.max_power = max_power
        
        # 这里的 Buffer 用于存储 grid，不需要梯度
        self.register_buffer('basis_bank', None)
        
        # --- 核心可学习参数 ---
        # 这是一个线性层，用于将 "基底库" 混合成 "物理算子权重图"
        # 输入维度: 1 (const) + max_power (pT) + 2*max_harmonics (phi) + 交互项
        # 我们先动态计算一下基底的总数，稍后初始化
        self.mixing_weights = None # 将在第一次 forward 时初始化，或手动指定维度

    def _build_basis_bank(self, H, W, device):
        if self.basis_bank is not None and self.basis_bank.shape[2:] == (H, W):
            return

        # 1. 构建坐标网格 (假设归一化到物理范围)
        # 假设 crop 后对应 pT < 0.17 GeV，归一化 pT 到 [0, 1] 方便数值稳定性
        y = torch.linspace(-1, 1, H, device=device)
        x = torch.linspace(-1, 1, W, device=device)
        gy, gx = torch.meshgrid(y, x)
        
        pt = torch.sqrt(gx**2 + gy**2) + 1e-6 # 避免 log(0)
        phi = torch.atan2(gy, gx)
        
        basis_list = []
        
        # --- Group A: 径向基底 (Radial Basis) ---
        basis_list.append(torch.ones_like(pt))      # 0阶: Count/Yield
        basis_list.append(torch.log(pt + 1e-6))     # Log pT (低 pT 敏感)
        for p in range(1, self.max_power + 1):
            basis_list.append(pt ** p)              # pT, pT^2, pT^3...
            
        # --- Group B: 角向基底 (Angular Basis) ---
        # 纯角度项 (对应 v2, v3 等流强，但不带 pT 权重)
        for h in range(1, self.max_harmonics + 1):
            basis_list.append(torch.cos(h * phi))
            basis_list.append(torch.sin(h * phi))
            
        # --- Group C: 耦合基底 (Coupled Basis) ---
        # 物理学中常用的 pT 依赖流: pT * cos(2phi), pT^2 * cos(2phi)
        # 这里为了简化，我们只添加一阶 pT 耦合
        for h in range(1, self.max_harmonics + 1):
            basis_list.append(pt * torch.cos(h * phi))
            basis_list.append(pt * torch.sin(h * phi))

        # 堆叠所有基底: (1, N_basis, H, W)
        self.basis_bank = torch.stack(basis_list, dim=0).unsqueeze(0)
        
        # 初始化混合层
        input_dim = self.basis_bank.shape[1]
        if self.mixing_weights is None:
            # 这是一个 (N_basis -> N_out) 的映射
            # 使用 1x1 Conv 实现像素级的线性组合
            self.mixer = nn.Conv2d(input_dim, self.num_features, kernel_size=1, bias=False).to(device)
            
            # 初始化技巧：让前几个特征直接对应明确的物理量 (Identity Init)
            # 例如: Feat 0 = Yield, Feat 1 = Mean pT, Feat 2 = v2_cos
            nn.init.orthogonal_(self.mixer.weight) # 保持正交性，鼓励学习不同的特征

    def forward(self, img, masks):
        """
        img: (B, 1, H, W)
        masks: (B, N_tasks, H, W)
        """
        B, N_tasks, H, W = masks.shape
        self._build_basis_bank(H, W, img.device)
        
        # 1. 生成算子权重图: (1, N_feat, H, W)
        weight_maps = self.mixer(self.basis_bank) 
        
        features_list = []
        
        # 预处理
        img = F.relu(img)
        
        for t in range(N_tasks):
            mask = masks[:, t:t+1] # (B, 1, H, W)
            
            # 有效图像区域
            masked_img = img * mask 
            
            # 计算总产额 (Yield)
            # [Fix 1] 彻底展平为 (B, 1)，避免后续广播错误
            total_yield = masked_img.sum(dim=(2, 3)) # (B, 1)
            safe_yield = torch.clamp(total_yield, min=1e-5)
            
            # 分子: (masked_img * weight_maps).sum
            # masked_img: (B, 1, H, W)
            # weight_maps: (1, N_feat, H, W)
            # Product -> (B, N_feat, H, W) -> Sum -> (B, N_feat)
            numerator = (masked_img * weight_maps).sum(dim=(2, 3)) 
            
            # [Fix 2] 直接进行 (B, N_feat) / (B, 1) 的除法
            # PyTorch 会自动将 (B, 1) 广播为 (B, N_feat)
            intensive_feat = numerator / safe_yield
            
            # [Fix 3] 构造 Log Yield，保持 (B, 1)
            log_yield = torch.log(total_yield + 1) # (B, 1)
            
            # 拼接: (B, N_feat) cat (B, 1) -> (B, N_feat + 1)
            task_feat = torch.cat([intensive_feat, log_yield], dim=1)
            features_list.append(task_feat)
            
        return torch.stack(features_list, dim=1)


class TaskHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        # 从纯线性 Linear 变为 MLP
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(32, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)
# ==========================================
# 模块 D: 整合模型 (Replacing MultiTaskResNetMLP)
# ==========================================

class PhysicalDiscoveryModel(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        
        # 1. [设定] 想要几个探测区域？
        # 比如 8 个区域。这样总特征数 = 8 * 16 = 128
        self.n_regions = 8 
        self.obs_dim_per_region = 16 # 你的 Adapter 默认输出维度
        
        # 2. [新增] 多区域生成器
        self.region_generator = LearnableMultiRegionGenerator(
            H=224, W=224, num_regions=self.n_regions
        )
        
        # 3. [保留] 物理算子适配器
        # 注意：Adapter 内部会遍历传入的 masks 的 dim=1
        # 所以它会自然地对 8 个区域分别计算物理量
        self.adapter = LearnablePhysicsOperatorAdapter(num_learnable_features=self.obs_dim_per_region)
        
        # 4. [修改] 任务头输入维度
        # 输入维度变成了：区域数 * 每个区域的特征数
        total_features = self.n_regions * self.obs_dim_per_region
        
        self.feature_norm = nn.BatchNorm1d(total_features)
        
        # Task Heads 现在的输入是 128 维 (8*16)
        self.head_skin = TaskHead(total_features, 3) 
        self.head_def = TaskHead(total_features, 2) 

    def forward(self, x, hard_routing=False):
        if x.shape[1] == 3:
            x = x.mean(dim=1, keepdim=True)
        B, C, H, W = x.shape
        
        # 1. 生成 N 个区域 Mask
        # regions: (B, 8, H, W)
        regions = self.region_generator(B)
        
        # 2. 提取物理特征
        # Adapter 会把 regions 当作 distinct masks 处理
        # feats: (B, 8, 16) -> (Batch, Num_Regions, Num_Phys_Features)
        feats = self.adapter(x, regions)
        
        # 3. [关键步骤] 展平特征 (Flatten)
        # 将空间维(Regions)和特征维(Phys)合并
        # (B, 8, 16) -> (B, 128)
        feats_flat = feats.reshape(B, -1)
        
        # 4. 归一化 & 噪声
        feats_norm = self.feature_norm(feats_flat)
        
        if self.training:
            # 加一点噪声防止过拟合
            noise = torch.randn_like(feats_norm) * 0.05
            feats_norm = feats_norm + noise

        # 5. 预测
        # MLP 现在同时看到了 Region 1 的 v2, Region 2 的 Mean pT...
        out_skin = self.head_skin(feats_norm)
        out_def  = self.head_def(feats_norm)
        
        # 返回 regions 用于画图：看看AI把这8个框放在了哪！
        return out_def, out_skin # , regions (如果是 inference 模式建议返回 regions)


def create_model(config, device='cpu'):
    """模型创建工厂函数"""
    mode = config['inference_mode']
    if mode == "PhysicalDiscoveryModel":
        return PhysicalDiscoveryModel(config)
    else:
        raise ValueError(f"不支持的模式: {mode}")


# ===================================================================
# 4. 多任务训练与验证函数（集成GradNorm）
# ===================================================================

def setup_discriminative_lr_multitask(model, config):
    """
    为 MultiTaskResNetMLP 模型设置分层学习率。
    - ResNet 骨干网络被分成多个部分，越底层的学习率越低。
    - 所有新添加的层（特征映射、共享层、任务头）被视为一个整体的"头部"，使用最高的"基础学习率"。
    """
    lr_conf = config['learning_rates']
    base_lr = lr_conf['base']
    # 层衰减率，例如 0.9，意味着每深入一层，学习率乘以0.9
    decay = lr_conf.get('layer_decay', 0.9)

    # 确保模型是正确的类型
    if not isinstance(model, MultiTaskResNetMLP):
        raise TypeError(f"此函数专为 MultiTaskResNetMLP 设计，但收到的模型类型为 {type(model)}")

    # 1. 识别出 ResNet 骨干网络
    resnet_backbone = model.feature_extractor.features

    # 2. 将 ResNet 骨干网络分层 (此分组适用于 ResNet18/34)
    #    - 早期层: conv1, bn1, relu, maxpool, layer1, layer2
    #    - 中期层: layer3
    #    - 后期层: layer4
    layer_groups = [
        list(resnet_backbone[0].parameters()) + list(resnet_backbone[1].parameters()) +
        list(resnet_backbone[4].parameters()) + list(resnet_backbone[5].parameters()),
        list(resnet_backbone[6].parameters()),
        list(resnet_backbone[7].parameters()),
    ]

    # 3. 识别出所有新添加的"头部"层
    #    这包括: 特征投射层, 共享MLP, 回归头, 分类头
    head_layers = list(model.feature_extractor.feature_proj.parameters()) + \
                  list(model.shared_layers.parameters()) + \
                  list(model.regression_head.parameters()) + \
                  list(model.classification_head.parameters())

    # 4. 构建用于优化器的参数组
    optimizer_params = [
        # 为骨干网络设置递减的学习率
        {'params': layer_groups[0], 'lr': base_lr * (decay ** 3)},
        {'params': layer_groups[1], 'lr': base_lr * (decay ** 2)},
        {'params': layer_groups[2], 'lr': base_lr * decay},
        # 为所有头部层设置基础学习率
        {'params': head_layers, 'lr': base_lr}
    ]

    print("差分学习率设置成功:")
    print(f"  - 头部 (新层) 学习率: {base_lr:.6f}")
    print(f"  - ResNet 后期层学习率: {base_lr * decay:.6f}")
    print(f"  - ResNet 中期层学习率: {base_lr * (decay ** 2):.6f}")
    print(f"  - ResNet 早期层学习率: {base_lr * (decay ** 3):.6f}")
    
    return optimizer_params


def train_epoch_multitask_gradnorm(model, train_loader, regression_criterion, classification_criterion, 
                                  optimizer, gradnorm_optimizer, gradnorm_trainer,feasibility_loss_fn, device, epoch, config):
    """使用GradNorm的多任务训练一个epoch"""
    model.train()
    gradnorm_trainer.task_weights.train()
    
    running_total_loss = 0.0
    running_reg_loss = 0.0
    running_cls_loss = 0.0
    running_gradnorm_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    # 记录权重变化
    weight_history = []
    
    for i, (inputs, reg_labels, cls_labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        reg_labels = reg_labels.to(device)
        cls_labels = cls_labels.to(device)
        
        
        # 清零梯度
        optimizer.zero_grad()
        gradnorm_optimizer.zero_grad()
        
        # 前向传播
        reg_outputs, cls_outputs = model(inputs)
        
        beta2_pred = reg_outputs[:, 0]
        beta3_pred = reg_outputs[:, 1]
        beta2_target = reg_labels[:, 0]
        beta3_target = reg_labels[:, 1]
        
        # 分别计算 Loss (注意：这里不需要 keep_dim=True，直接求标量均值)
        # 假设 regression_criterion 是 MSELoss()
        loss_beta2 = regression_criterion(beta2_pred, beta2_target)
        loss_beta3 = regression_criterion(beta3_pred, beta3_target)
        cls_loss = classification_criterion(cls_outputs, cls_labels)
        
        # 堆叠成 3 个任务的 Loss 向量
        task_losses = torch.stack([loss_beta2, loss_beta3, cls_loss])
        reg_loss = loss_beta2+loss_beta3
        # GradNorm 计算 (它现在会返回加权后的总和)
        weighted_loss, gradnorm_loss = gradnorm_trainer.compute_grad_norm(task_losses)
        
        geo_reg_loss, repulsion_loss, size_loss = feasibility_loss_fn(model)


        # 总损失 = 加权任务损失 + GradNorm正则化损失
        gradnorm_weight = config.get('gradnorm', {}).get('weight', 0.1)
        total_loss = weighted_loss + gradnorm_weight * gradnorm_loss + geo_reg_loss
        
        # 反向传播和优化
        total_loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(gradnorm_trainer.task_weights.parameters(), max_norm=1.0)
        
        # 更新模型参数和任务权重
        optimizer.step()
        gradnorm_optimizer.step()
        
        # 重新规范化权重
        gradnorm_trainer.task_weights.renormalize()
        
        # 统计
        running_total_loss += total_loss.item()
        running_reg_loss += reg_loss.item()
        running_cls_loss += cls_loss.item()
        running_gradnorm_loss += gradnorm_loss.item()
        
        # 分类准确率
        _, predicted = cls_outputs.max(1)
        correct = predicted.eq(cls_labels).sum().item()
        correct_predictions += correct
        total_samples += cls_labels.size(0)
        
        # 记录权重
        current_weights = gradnorm_trainer.get_current_weights()
        weight_history.append(current_weights)
    
    # 计算平均值
    avg_total_loss = running_total_loss / len(train_loader)
    avg_reg_loss = running_reg_loss / len(train_loader)
    avg_cls_loss = running_cls_loss / len(train_loader)
    avg_gradnorm_loss = running_gradnorm_loss / len(train_loader)
    train_acc = 100. * correct_predictions / total_samples if total_samples > 0 else 0.0
    
    # 获取最终权重
    final_weights = gradnorm_trainer.get_current_weights()
    
    print(f"Epoch {epoch} - 训练完成:")
    print(f"  总损失: {avg_total_loss:.4f} | 回归损失: {avg_reg_loss:.4f} | 分类损失: {avg_cls_loss:.4f}")
    print(f"  GradNorm损失: {avg_gradnorm_loss:.4f} | 分类准确率: {train_acc:.2f}%")
    print(f"  任务权重 - 回归: {final_weights[0]:.4f}, 分类: {final_weights[1]:.4f}")
    
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
                                     gradnorm_trainer,feasibility_loss_fn, device, epoch):
    """
    使用GradNorm权重的多任务验证一个epoch (3任务模式: Beta2, Beta3, Classification)
    """
    model.eval()
    gradnorm_trainer.task_weights.eval()
    
    # 初始化统计变量
    running_total_loss = 0.0
    running_beta2_loss = 0.0  # 新增
    running_beta3_loss = 0.0  # 新增
    running_cls_loss = 0.0
    
    all_reg_preds, all_reg_labels = [], []
    all_cls_preds, all_cls_labels = [], []
    stats = val_loader.dataset.stats
    # 记得把 numpy/cpu tensor 转到 GPU 上进行计算
    mean_tensor = stats['mean'].to(device)
    std_tensor = stats['std'].to(device)
    # 获取当前任务权重 (现在应该是 3 个值)
    current_weights = gradnorm_trainer.get_current_weights()
    # 确保权重数量正确，方便调试
    if len(current_weights) != 3:
        print(f"警告: GradNorm权重数量为 {len(current_weights)}，预期为 3 (Beta2, Beta3, Class)")
        # 如果还是2个，暂时兼容处理（但这说明Main函数没改对）
        w_beta2, w_beta3, w_cls = (current_weights[0], current_weights[0], current_weights[1]) if len(current_weights)==2 else current_weights
    else:
        w_beta2, w_beta3, w_cls = current_weights
    
    with torch.no_grad():
        for inputs, reg_labels, cls_labels in val_loader:
            inputs = inputs.to(device)
            reg_labels = reg_labels.to(device)
            cls_labels = cls_labels.to(device)
            
            # 前向传播
            reg_outputs, cls_outputs = model(inputs)
            
            # --- 修改部分开始: 拆分回归任务 ---
            
            # 提取 Beta2 和 Beta3 的预测值与真实值
            beta2_pred = reg_outputs[:, 0]
            beta3_pred = reg_outputs[:, 1]
            beta2_target = reg_labels[:, 0]
            beta3_target = reg_labels[:, 1]
            
            # 分别计算损失 (Huber Loss 支持 1D 输入)
            loss_beta2 = regression_criterion(beta2_pred, beta2_target)
            loss_beta3 = regression_criterion(beta3_pred, beta3_target)
            loss_cls = classification_criterion(cls_outputs, cls_labels)
            

            geo_reg_loss, repulsion_loss, size_loss = feasibility_loss_fn(model)

            # 使用 3 个 GradNorm 权重计算加权总损失
            # 注意：这里的顺序必须和训练循环中 torch.stack 的顺序一致
            total_loss = w_beta2 * loss_beta2 + w_beta3 * loss_beta3 + w_cls * loss_cls + geo_reg_loss
            

            # 记录损失
            running_total_loss += total_loss.item()
            running_beta2_loss += loss_beta2.item()
            running_beta3_loss += loss_beta3.item()
            running_cls_loss += loss_cls.item()
            
            # --- 修改部分结束 ---
            

            mean = val_loader.dataset.stats['mean'].to(device)
            std = val_loader.dataset.stats['std'].to(device)
            
            # 还原预测值: pred * std + mean
            pred_real = reg_outputs * std + mean
            # 还原标签值: label * std + mean
            target_real = reg_labels * std + mean
            
            # --- 3. 收集数据用于计算 R2 ---
            # 存入列表的是 numpy 数组，且是还原后的真实值
            all_reg_preds.extend(pred_real.cpu().numpy())
            all_reg_labels.extend(target_real.cpu().numpy())

            # 收集预测结果 (用于计算 R2, Acc 等指标)
            #all_reg_preds.extend(reg_outputs.cpu().numpy())
            #all_reg_labels.extend(reg_labels.cpu().numpy())
            
            _, predicted = cls_outputs.max(1)
            all_cls_preds.extend(predicted.cpu().numpy())
            all_cls_labels.extend(cls_labels.cpu().numpy())
    
    # 计算平均损失
    num_batches = len(val_loader)
    avg_total_loss = running_total_loss / num_batches
    avg_beta2_loss = running_beta2_loss / num_batches
    avg_beta3_loss = running_beta3_loss / num_batches
    avg_cls_loss = running_cls_loss / num_batches
    # 为了兼容之前的日志格式，算一个平均回归损失
    avg_reg_loss = (avg_beta2_loss + avg_beta3_loss) / 2 
    
    # --- 以下指标计算部分基本不需要变，因为是基于 numpy 数组计算的 ---
    
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
    
    print(f"验证完成 (3-Task GradNorm):")
    print(f"  总损失: {avg_total_loss:.4f}")
    print(f"  损失详情 - Beta2: {avg_beta2_loss:.4f} | Beta3: {avg_beta3_loss:.4f} | 分类: {avg_cls_loss:.4f}")
    print(f"  Beta2 - R²: {r2_beta2:.6f}, MSE: {mse_beta2:.6f}")
    print(f"  Beta3 - R²: {r2_beta3:.6f}, MSE: {mse_beta3:.6f}")
    print(f"  分类准确率: {val_acc:.2f}%")
    print(f"  当前权重 - w_Beta2: {w_beta2:.4f}, w_Beta3: {w_beta3:.4f}, w_Cls: {w_cls:.4f}")
    
    metrics = {
        'total_loss': avg_total_loss,
        'regression_loss': avg_reg_loss, # 仅作参考
        'beta2_loss': avg_beta2_loss,    # 新增
        'beta3_loss': avg_beta3_loss,    # 新增
        'classification_loss': avg_cls_loss,
        'regression_mse': reg_mse,
        'regression_mae': reg_mae,
        'mse_beta2': mse_beta2, 'mse_beta3': mse_beta3,
        'mae_beta2': mae_beta2, 'mae_beta3': mae_beta3,
        'r2_beta2': r2_beta2, 'r2_beta3': r2_beta3,
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
    
    # 1. 设置损失函数 (保持不变)
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
    
    # 2. 设置主模型优化器 [MODIFIED]
    lr = config['learning_rates']['base']
    weight_decay = config['weight_decay']


    # [NEW] 如果是新的 PhysicalRoutingModel，使用简单配置
    if hasattr(model, 'router'): 
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        print(f"检测到物理路由模型，使用标准 Adam 优化器 (LR={lr})")
        
    else:
        generator_params = list(map(id, model.region_generator.parameters()))
        base_params = filter(lambda p: id(p) not in generator_params, model.parameters())

        optimizer = torch.optim.AdamW([
            # MLP 和 Adapter 使用正常的学习率 (例如 1e-3)
            {'params': base_params, 'lr': lr}, 
            
            # [关键] Mask 生成器使用 1/10 的学习率
            # 让它慢慢地寻找全局最优位置，而不是在 Batch 之间反复横跳
            {'params': model.region_generator.parameters(), 'lr': lr * 0.01} 
        ], weight_decay=1e-4)
        #optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # 3. GradNorm 权重优化器 (保持不变)
    gradnorm_lr = config.get('gradnorm', {}).get('lr', 0.025)
    gradnorm_optimizer = optim.Adam(gradnorm_trainer.task_weights.parameters(), lr=gradnorm_lr)
    
    return optimizer, gradnorm_optimizer, regression_criterion, classification_criterion


def plot_multitask_results_gradnorm(reg_predictions, reg_labels, cls_predictions, cls_labels, 
                                   output_path, epoch, val_metrics, weight_history=None):
    """绘制多任务结果（GradNorm版本）"""
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
        axes[0, 3].set_xlabel('Training Steps')
        axes[0, 3].set_ylabel('Task Weight')
        axes[0, 3].set_title('GradNorm Task Weight Evolution')
        axes[0, 3].legend()
        axes[0, 3].grid(True, alpha=0.3)
    else:
        # 显示当前权重信息
        current_weights = val_metrics.get('task_weights', [1.0, 1.0])
        axes[0, 3].text(0.5, 0.5, f'Current Task Weights:\n\nRegression: {current_weights[0]:.4f}\nClassification: {current_weights[1]:.4f}', 
                       transform=axes[0, 3].transAxes, fontsize=14,
                       horizontalalignment='center', verticalalignment='center',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        axes[0, 3].set_title('GradNorm Task Weights')
        axes[0, 3].axis('off')
    
    # 第二行：分类结果
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
    axes[1, 2].text(0.5, 0.5, f'Classification Accuracy: {val_metrics["classification_accuracy"]:.2f}%\n\n'
                               f'Regression MSE: {val_metrics["regression_mse"]:.6f}\n'
                               f'Regression MAE: {val_metrics["regression_mae"]:.6f}\n\n'
                               f'Total Loss: {val_metrics["total_loss"]:.4f}\n'
                               f'Regression Loss: {val_metrics["regression_loss"]:.4f}\n'
                               f'Classification Loss: {val_metrics["classification_loss"]:.4f}', 
                   transform=axes[1, 2].transAxes, fontsize=12,
                   horizontalalignment='center', verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    axes[1, 2].set_title('Multi-Task Performance Summary')
    axes[1, 2].axis('off')
    
    # GradNorm信息
    gradnorm_info = f'GradNorm Configuration:\n\n'
    if 'gradnorm_loss' in val_metrics:
        gradnorm_info += f'GradNorm Loss: {val_metrics["gradnorm_loss"]:.4f}\n'
    current_weights = val_metrics.get('task_weights', [1.0, 1.0])
    gradnorm_info += f'Weight Ratio: {current_weights[0]/current_weights[1]:.3f}\n\n'
    gradnorm_info += 'Task Balancing:\nAutomatically adjusted\nvia gradient norms'
    
    axes[1, 3].text(0.5, 0.5, gradnorm_info, 
                   transform=axes[1, 3].transAxes, fontsize=11,
                   horizontalalignment='center', verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[1, 3].set_title('GradNorm Status')
    axes[1, 3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_best_multitask_outputs_gradnorm(reg_predictions, reg_labels, cls_predictions, cls_labels, 
                                        output_dir, val_metrics, weight_history=None):
    """保存最佳多任务模型的输出数据和图片（GradNorm版本）"""
    # 保存图片
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
    
    # 保存数据到npz文件
    npz_path = output_dir / 'best_multitask_gradnorm_output.npz'
    save_data = {
        'reg_predictions': reg_predictions,
        'reg_labels': reg_labels,
        'cls_predictions': cls_predictions,
        'cls_labels': cls_labels,
    }
    
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
    """主训练函数（GradNorm版本）"""
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
        train_dataset = MultiTaskInterferenceDataset(
            Path(config['data_path']) / 'train', 
            config=config, 
            split='train',
            stats=None  # 训练集让它自己算
        )
        
        # 2. 获取训练集的统计量
        train_stats = train_dataset.stats
        
        # 3. 创建验证数据集，并传入训练集的统计量
        val_dataset = MultiTaskInterferenceDataset(
            Path(config['data_path']) / 'val', 
            config=config, 
            split='val',
            stats=train_stats  # 验证集必须用训练集的标准
        )
        
        # 创建模型
        model = create_model(config, device=device)
        model.to(device)
        
        feasibility_loss_fn = ExperimentalFeasibilityLoss(lambda_sep=0.1, lambda_size=0.05).to(device)
        # 创建GradNorm训练器
        gradnorm_config = config.get('gradnorm', {})
        gradnorm_alpha = gradnorm_config.get('alpha', 1.5)
        gradnorm_trainer = GradNormTrainer(model, num_tasks=3, alpha=gradnorm_alpha)
        gradnorm_trainer.task_weights.to(device)
        
        # 创建优化器和损失函数
        optimizer, gradnorm_optimizer, regression_criterion, classification_criterion = \
            create_multitask_optimizer_and_criterion_gradnorm(model, gradnorm_trainer, config)
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # 最佳模型跟踪
        best_total_loss = float('inf')
        best_r2_score = -float('inf')  # R2可能为负值，所以用负无穷
        best_model_path = checkpoints_dir / 'best_multitask_gradnorm_model.pth'
        all_weight_history = []
        #best_model_path = checkpoints_dir / 'best_multitask_gradnorm_model.pth'
        #all_weight_history = []
        
        # 开始训练
        logging.info(f"开始多任务GradNorm训练...")
        logging.info(f"GradNorm配置:")
        logging.info(f"  - Alpha: {gradnorm_alpha}")
        logging.info(f"  - GradNorm学习率: {gradnorm_config.get('lr', 0.025)}")
        logging.info(f"  - GradNorm权重: {gradnorm_config.get('weight', 0.1)}")
        

        
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
                optimizer, gradnorm_optimizer, gradnorm_trainer,feasibility_loss_fn, device, epoch, config
            )
            if config['inference_mode'] == 'physical_routing_poc' and hasattr(model, 'last_P'):
            # 获取 P 矩阵 (3x3)
                P = model.last_P.cpu().numpy()
            
            # 格式化打印
                print(f"\n[Epoch {epoch}] Router Assignment Matrix (Probability):")
                print(f"{'':>12} {'Mask0(Bright)':>12} {'Mask1(Dark)':>12} {'Mask2(Noise)':>12}")
                row_names = ["Task0(Skin)", "Task1(Def )", "Task2(Trash)"]
            
                #for i in range(3):
                #    row_str = f"{row_names[i]:>12} "
                #    for j in range(3):
                #        val = P[i, j]
                        # 如果值很大(>0.8)显示为绿色/高亮，方便人眼观察
                        # 这里简单用星号标记最大值
                #        marker = "*" if val > 0.8 else " "
                #        row_str += f"{val:>10.4f}{marker} "
                #    print(row_str)
                
                # 打印温度
                #print("Feature Importance (Linear Layer Weights):")
                # 更新了算子名称列表
                #op_names = [
                #    "Mean_Dens", "Std_Dens", "Yield", "Max_Dens", 
                #    "Mean_pT",   "Std_pT", 
                #    "v2_Flow",   "v3_Flow",  "v2_HighPt"
                #]
                
                # Skin Head 权重
                #w_skin = model.head_skin.weight.detach().abs().mean(dim=0).cpu().numpy()
                #top_skin_idx = w_skin.argmax()
                #print(f"  Skin Head focuses on: {op_names[top_skin_idx]} (val={w_skin.max():.4f})")
                
                # Def Head 权重
                #w_def = model.head_def.weight.detach().abs().mean(dim=0).cpu().numpy()
                #top_def_idx = w_def.argmax()
                #print(f"  Def  Head focuses on: {op_names[top_def_idx]} (val={w_def.max():.4f})")
                #print("-" * 60)
        # =======================================================

        # 2. 验证
            val_total_loss, val_metrics = validate_epoch_multitask_gradnorm(
                model, val_loader, regression_criterion, classification_criterion, 
                gradnorm_trainer, feasibility_loss_fn,device, epoch
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
                writer.add_scalar('Loss/Train_GradNorm', train_metrics['gradnorm_loss'], epoch)
                writer.add_scalar('Loss/Val_Total', val_total_loss, epoch)
                writer.add_scalar('Loss/Val_Regression', val_metrics['regression_loss'], epoch)
                writer.add_scalar('Loss/Val_Classification', val_metrics['classification_loss'], epoch)
                
                # 准确率
                writer.add_scalar('Accuracy/Train', train_metrics['train_accuracy'], epoch)
                writer.add_scalar('Accuracy/Validation', val_metrics['classification_accuracy'], epoch)
                
                # 回归指标
                writer.add_scalar('Regression/MSE', val_metrics['regression_mse'], epoch)
                writer.add_scalar('Regression/MAE', val_metrics['regression_mae'], epoch)
                writer.add_scalar('Regression/R2_Beta2', val_metrics['r2_beta2'], epoch)
                writer.add_scalar('Regression/R2_Beta3', val_metrics['r2_beta3'], epoch)
                
                # 任务权重
                current_weights = train_metrics['task_weights']
                writer.add_scalar('GradNorm/Weight_Regression', current_weights[0], epoch)
                writer.add_scalar('GradNorm/Weight_Classification', current_weights[1], epoch)
                writer.add_scalar('GradNorm/Weight_Ratio', current_weights[0]/current_weights[1], epoch)
                
                # 学习率
                writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
            
            # 绘制多任务结果（定期）
            if config['validation']['plot_predictions'] and epoch % config['validation']['plot_interval'] == 0:
                plot_path = plots_dir / f'multitask_gradnorm_results_epoch_{epoch}.png'
                # 合并所有权重历史用于绘图
                combined_weight_history = np.vstack(all_weight_history) if all_weight_history else None
                plot_multitask_results_gradnorm(
                    val_metrics['reg_predictions'], 
                    val_metrics['reg_labels'],
                    val_metrics['cls_predictions'],
                    val_metrics['cls_labels'],
                    plot_path, 
                    epoch, 
                    val_metrics,
                    weight_history=combined_weight_history
                )
            current_r2 = (0.8*val_metrics['r2_beta2'] + val_metrics['r2_beta3']) / 2
            # 保存最佳模型
            if current_r2 > best_r2_score:
                best_total_loss = val_total_loss
                best_r2_score = current_r2 
                best_total_loss = val_total_loss
                logging.info(f"新的最佳模型! 验证总损失: {best_total_loss:.6f}")
                
                if config['save_options']['save_best_only']:
                    save_payload = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'gradnorm_weights_state_dict': gradnorm_trainer.task_weights.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'gradnorm_optimizer_state_dict': gradnorm_optimizer.state_dict(),
                        'best_total_loss': best_total_loss,
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
                        output_dir, 
                        val_metrics,
                        weight_history=combined_weight_history
                    )
            
            # 输出训练日志
            logging.info(f"Epoch {epoch} 总结:")
            logging.info(f"  训练 - 总损失: {train_metrics['total_loss']:.4f}, 回归: {train_metrics['regression_loss']:.4f}, "
                        f"分类: {train_metrics['classification_loss']:.4f}, GradNorm: {train_metrics['gradnorm_loss']:.4f}")
            logging.info(f"  训练准确率: {train_metrics['train_accuracy']:.2f}%")
            logging.info(f"  验证 - 总损失: {val_total_loss:.4f}, 回归MSE: {val_metrics['regression_mse']:.6f}, "
                        f"分类准确率: {val_metrics['classification_accuracy']:.2f}%")
            current_weights = val_metrics['task_weights']
            logging.info(f"  当前任务权重 - 回归: {current_weights[0]:.4f}, 分类: {current_weights[1]:.4f}")
        
        if writer:
            writer.close()
        logging.info(f"多任务GradNorm训练完成! 最佳验证总损失: {best_total_loss:.6f}")
        
    finally:
        # 确保恢复标准输出并关闭日志文件
        sys.stdout = custom_logger.terminal
        custom_logger.close()


if __name__ == '__main__':
    main()