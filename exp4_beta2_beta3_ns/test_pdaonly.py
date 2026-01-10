import os
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
import sys

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, confusion_matrix, mean_squared_error, classification_report

# 导入训练程序中的类
from train2 import (
    MultiTaskInterferenceDataset, 
    create_model, 
    set_seed, 
    get_device
)

# =========================================================================
# 1. 全局绘图风格设置 (出版级)
# =========================================================================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white"
})

# =========================================================================
# 2. PDA 分析核心类 (保持不变)
# =========================================================================
class ConditionalPredictionDifferenceAnalyzer:
    def __init__(self, model, patch_size=8, stride=4, num_samples=10, 
                 sampling_mode='gaussian', context_size=3):
        self.model = model
        self.patch_size = patch_size
        self.stride = stride
        self.num_samples = num_samples
        self.sampling_mode = sampling_mode
        self.context_size = context_size
        
    def _get_context_stats(self, image, y, y_end, x, x_end):
        _, C, H, W = image.shape
        ctx_y_start = max(0, y - self.context_size)
        ctx_y_end = min(H, y_end + self.context_size)
        ctx_x_start = max(0, x - self.context_size)
        ctx_x_end = min(W, x_end + self.context_size)
        context = image[:, :, ctx_y_start:ctx_y_end, ctx_x_start:ctx_x_end].clone()
        
        means = [context[0, c].mean().item() for c in range(C)]
        stds = [context[0, c].std().item() + 1e-6 for c in range(C)]
        return torch.tensor(means), torch.tensor(stds)
    
    def _sample_patch_gaussian(self, image, y, y_end, x, x_end):
        _, C, _, _ = image.shape
        mean, std = self._get_context_stats(image, y, y_end, x, x_end)
        sampled = image.clone()
        for c in range(C):
            noise = torch.randn(y_end - y, x_end - x, device=image.device)
            sampled[0, c, y:y_end, x:x_end] = mean[c] + std[c] * noise
        return sampled

    def generate_pda(self, input_tensor, task_type='classification', 
                    target_class=None, regression_dim=None, batch_size=32):
        self.model.eval()
        device = input_tensor.device
        _, C, H, W = input_tensor.shape
        
        with torch.no_grad():
            reg_output, cls_output = self.model(input_tensor)
            
        if task_type == 'classification':
            target_class = target_class or torch.argmax(cls_output, dim=1).item()
            baseline_pred = cls_output[0, target_class].item()
        else:
            regression_dim = regression_dim or 0
            baseline_pred = reg_output[0, regression_dim].item()
        
        importance_sum = np.zeros((H, W), dtype=np.float32)
        count_map = np.zeros((H, W), dtype=np.float32)
        all_samples = []
        
        for y in range(0, H, self.stride):
            for x in range(0, W, self.stride):
                x_end = min(x + self.patch_size, W)
                y_end = min(y + self.patch_size, H)
                for _ in range(self.num_samples):
                    all_samples.append(self._sample_patch_gaussian(input_tensor, y, y_end, x, x_end))
        
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
        
        idx = 0
        for y in range(0, H, self.stride):
            for x in range(0, W, self.stride):
                x_end = min(x + self.patch_size, W)
                y_end = min(y + self.patch_size, H)
                patch_preds = all_predictions[idx:idx+self.num_samples]
                avg_diff = np.mean([abs(baseline_pred - p) for p in patch_preds])
                importance_sum[y:y_end, x:x_end] += avg_diff
                count_map[y:y_end, x:x_end] += 1
                idx += self.num_samples
        
        heatmap = importance_sum / (count_map + 1e-8)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap

# =========================================================================
# 3. 数据集类 (复用)
# =========================================================================
class SpecificFilesDataset(MultiTaskInterferenceDataset):
    def __init__(self, root_dir, config, target_files):
        super().__init__(root_dir, config, split='test')
        self.file_paths = []
        # 严格按照 target_files 列表顺序加载
        for target_name in target_files:
            target_path = Path(root_dir) / target_name
            if target_path.exists():
                self.file_paths.append(target_path)
            else:
                print(f"Warning: 指定的文件不存在 {target_path}")
        
    def __getitem__(self, idx):
        image_tensor, regression_label, classification_label = super().__getitem__(idx)
        file_path = str(self.file_paths[idx])
        processed_image = image_tensor.clone()
        return image_tensor, regression_label, classification_label, file_path, processed_image

# =========================================================================
# 4. 绘图函数
# =========================================================================
def save_pda_paper_style(heatmap, processed_image, output_path, config, global_plot_index):
    # --- A. 数据准备 ---
    if isinstance(processed_image, torch.Tensor):
        img_tensor = processed_image.cpu()
    else:
        img_tensor = torch.from_numpy(processed_image)
        
    img_display = img_tensor.numpy().transpose(2, 1, 0)
    # ImageNet 反标准化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_display = std * img_display + mean
    img_display = np.clip(img_display, 0, 1)
    
    height, width = img_display.shape[:2]
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.cpu().numpy()
    
    # 翻转逻辑
    heatmap_resized = cv2.resize(heatmap, (width, height))
    img_for_plot = np.flipud(img_display)
    heatmap_for_plot = np.flipud(heatmap_resized)
    heatmap_high_res = cv2.resize(heatmap_for_plot, (heatmap_for_plot.shape[1], heatmap_for_plot.shape[0]), interpolation=cv2.INTER_CUBIC)
    # --- B. 绘图 ---
    fig, ax = plt.subplots(figsize=(6, 5), facecolor='white')
    p_limit = config['interpretability']['pda'].get('p_limit', 0.17)
    extent = [-p_limit, p_limit, -p_limit, p_limit]
    
    ax.imshow(img_for_plot, extent=extent, origin='lower')
    
    alpha = config['interpretability']['pda'].get('alpha', 0.4)
    cmap = config['interpretability']['pda'].get('colormap', 'jet')
    
    im = ax.imshow(heatmap_for_plot, extent=extent, origin='lower',
                   cmap=cmap, alpha=alpha, vmin=0, vmax=1)
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.yaxis.set_tick_params(color='black')
    cbar.outline.set_edgecolor('black')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='black', fontsize=12)

    # --- C. 样式精修 ---
    ax.set_xlim(-p_limit, p_limit)
    ax.set_ylim(-p_limit, p_limit)
    
    # 标题生成: (a), (b)...
    title_char = chr(97 + global_plot_index) 
    title_text = f"({title_char})"
    ax.set_title(title_text, color='black', weight='bold', fontsize=18, loc='center')

    ax.set_xlabel(r'$p_x$ (GeV/c)', color='black', fontsize=14)
    ax.set_ylabel(r'$p_y$ (GeV/c)', color='black', fontsize=14)
    ax.tick_params(direction='in', colors='black', which='both', top=True, right=True, labelsize=12)
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  ✅ Saved [{title_text}]: {output_path.name}")

# =========================================================================
# 5. 核心逻辑控制
# =========================================================================
def process_task_queue(model, device, config, output_dir):
    """
    分别处理回归和分类列表，确保每个文件夹内的标题从 (a) 开始重新计数
    """
    pda_config = config['interpretability']['pda']
    data_path = Path(config['test_config'].get('test_data_path'))
    pda_root = output_dir / pda_config['output_dir']
    
    # 初始化分析器
    analyzer = ConditionalPredictionDifferenceAnalyzer(
        model,
        patch_size=pda_config.get('patch_size', 8),
        stride=pda_config.get('stride', 4),
        num_samples=pda_config.get('num_conditional_samples', 10),
        sampling_mode=pda_config.get('sampling_mode', 'gaussian')
    )

    # ==========================
    # Phase 1: 回归任务 (Regression)
    # ==========================
    if pda_config['task_specific']['regression']:
        reg_files = pda_config.get('target_files_regression', [])
        if reg_files:
            print(f"\n>>> 开始处理回归任务 PDA (共 {len(reg_files)} 个文件)...")
            save_dir = pda_root / 'regression'
            save_dir.mkdir(parents=True, exist_ok=True)
            
            dataset = SpecificFilesDataset(data_path, config, reg_files)
            loader = DataLoader(dataset, batch_size=1, shuffle=False)
            
            counter = 0 # 重置计数器
            
            for inputs, _, _, file_path_tuple, processed_images in tqdm(loader, desc="Regression PDA"):
                inputs = inputs.to(device)
                processed_image = processed_images[0]
                file_name = Path(file_path_tuple[0]).stem
                
                # Beta 2
                heatmap_b2 = analyzer.generate_pda(inputs, task_type='regression', regression_dim=0,
                                                   batch_size=pda_config.get('internal_batch_size', 32))
                save_name_b2 = save_dir / f"{file_name}_beta2.png"
                save_pda_paper_style(heatmap_b2, processed_image, save_name_b2, config, counter)
                counter += 1 # (a) -> (b)
                
                # Beta 3
                heatmap_b3 = analyzer.generate_pda(inputs, task_type='regression', regression_dim=1,
                                                   batch_size=pda_config.get('internal_batch_size', 32))
                save_name_b3 = save_dir / f"{file_name}_beta3.png"
                save_pda_paper_style(heatmap_b3, processed_image, save_name_b3, config, counter)
                counter += 1 # (b) -> (c)
        else:
            print("Config 中未找到 'target_files_regression' 或为空。")

    # ==========================
    # Phase 2: 分类任务 (Classification)
    # ==========================
    if pda_config['task_specific']['classification']:
        cls_files = pda_config.get('target_files_classification', [])
        if cls_files:
            print(f"\n>>> 开始处理分类任务 PDA (共 {len(cls_files)} 个文件)...")
            save_dir = pda_root / 'classification'
            save_dir.mkdir(parents=True, exist_ok=True)
            
            dataset = SpecificFilesDataset(data_path, config, cls_files)
            loader = DataLoader(dataset, batch_size=1, shuffle=False)
            
            counter = 0 # 重置计数器，分类图从 (a) 开始
            
            for inputs, _, _, file_path_tuple, processed_images in tqdm(loader, desc="Classification PDA"):
                inputs = inputs.to(device)
                processed_image = processed_images[0]
                file_name = Path(file_path_tuple[0]).stem
                
                heatmap = analyzer.generate_pda(inputs, task_type='classification',
                                                batch_size=pda_config.get('internal_batch_size', 32))
                save_name = save_dir / f"{file_name}_cls.png"
                save_pda_paper_style(heatmap, processed_image, save_name, config, counter)
                counter += 1 # (a) -> (b)
        else:
            print("Config 中未找到 'target_files_classification' 或为空。")

def calculate_metrics_only(model, loader, device):
    """仅计算统计指标"""
    print("\n" + "="*50)
    print("Computing Global Metrics (R2 & Confusion Matrix)...")
    print("="*50)
    model.eval()
    all_reg, all_reg_lbl = [], []
    all_cls, all_cls_lbl = [], []
    
    with torch.no_grad():
        for inputs, reg_lbl, cls_lbl in tqdm(loader, desc="Evaluating"):
            inputs = inputs.to(device)
            reg_out, cls_out = model(inputs)
            all_reg.extend(reg_out.cpu().numpy())
            all_reg_lbl.extend(reg_lbl.cpu().numpy())
            all_cls.extend(cls_out.max(1)[1].cpu().numpy())
            all_cls_lbl.extend(cls_lbl.cpu().numpy())
            
    r2_b2 = r2_score(np.array(all_reg_lbl)[:,0], np.array(all_reg)[:,0])
    r2_b3 = r2_score(np.array(all_reg_lbl)[:,1], np.array(all_reg)[:,1])
    cm = confusion_matrix(all_cls_lbl, all_cls)
    
    print("-" * 40)
    print(f"Beta2 R²: {r2_b2:.4f}")
    print(f"Beta3 R²: {r2_b3:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("-" * 40)

def main():
    with open('config_pda.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    set_seed(config['seed'])
    device = get_device(config['device'])
    output_dir = Path(config['output_dir']) / config['experiment_name'] / 'test_results'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载模型
    ckpt = torch.load(config['test_config']['model_checkpoint'], map_location=device)
    model = create_model(config, device=device)
    model.to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    
    # 1. 计算全局指标 (使用全量测试集)
    test_path = Path(config['test_config'].get('test_data_path'))
    full_ds = MultiTaskInterferenceDataset(test_path, config, split='test')
    full_loader = DataLoader(full_ds, batch_size=64, shuffle=False, num_workers=4)
    calculate_metrics_only(model, full_loader, device)
    
    # 2. 执行特定列表的 PDA 分析
    if config['interpretability']['pda']['enabled']:
        process_task_queue(model, device, config, output_dir)
        
    print("\nDone.")

if __name__ == '__main__':
    main()