import os
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, confusion_matrix, mean_squared_error, mean_absolute_error, classification_report

import torch
from torch.utils.data import DataLoader

# 导入训练程序中的类
from train3 import (
    MultiTaskInterferenceDataset, 
    MultiTaskResNetMLP, 
    CustomLogger,
    create_model, 
    set_seed, 
    get_device,
    DataPreprocessor
)

# ==========================================
# 绘图风格设置 (保持与原来一致)
# ==========================================
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.linewidth": 1.5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": False, # 右侧坐标轴由 twinx 处理
    "xtick.major.size": 6,
    "ytick.major.size": 6,
})

# ==========================================
# 辅助类与函数
# ==========================================

def load_model_from_checkpoint(checkpoint_path, config, device):
    """从checkpoint加载模型"""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"模型checkpoint不存在: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = create_model(config, device=device)
    model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"成功加载模型权重，来自epoch {checkpoint['epoch']}")
    return model

def test_model(model, test_loader, device):
    """测试模型并收集预测结果"""
    model.eval()
    
    all_reg_preds, all_reg_labels = [], []
    all_cls_preds, all_cls_labels = [], []
    
    with torch.no_grad():
        for batch_idx, (inputs, reg_labels, cls_labels, file_paths, processed_images) in enumerate(test_loader):
            inputs = inputs.to(device)
            reg_labels = reg_labels.to(device)
            cls_labels = cls_labels.to(device)
            
            # 前向传播
            reg_outputs, cls_outputs = model(inputs)
            
            # 收集结果
            all_reg_preds.extend(reg_outputs.cpu().numpy())
            all_reg_labels.extend(reg_labels.cpu().numpy())
            
            _, predicted = cls_outputs.max(1)
            all_cls_preds.extend(predicted.cpu().numpy())
            all_cls_labels.extend(cls_labels.cpu().numpy())
    
    return {
        'reg_predictions': np.array(all_reg_preds),
        'reg_labels': np.array(all_reg_labels),
        'cls_predictions': np.array(all_cls_preds),
        'cls_labels': np.array(all_cls_labels),
    }

def evaluate_results(results):
    """计算指标"""
    reg_preds = results['reg_predictions']
    reg_labels = results['reg_labels']
    cls_preds = results['cls_predictions']
    cls_labels = results['cls_labels']
    
    # R2 Scores
    r2_beta2 = r2_score(reg_labels[:, 0], reg_preds[:, 0])
    r2_beta3 = r2_score(reg_labels[:, 1], reg_preds[:, 1])
    avg_r2 = (r2_beta2 + r2_beta3) / 2
    
    # Accuracy
    test_acc = 100. * np.sum(cls_preds == cls_labels) / len(cls_labels)
    
    return {
        'r2_beta2': r2_beta2,
        'r2_beta3': r2_beta3,
        'avg_r2': avg_r2,
        'classification_accuracy': test_acc
    }

class TestMultiTaskDataset(MultiTaskInterferenceDataset):
    """测试用的多任务数据集"""
    def __getitem__(self, idx):
        image_tensor, regression_label, classification_label = super().__getitem__(idx)
        file_path = str(self.file_paths[idx])
        processed_image = image_tensor.clone()
        return image_tensor, regression_label, classification_label, file_path, processed_image

# ==========================================
# 核心绘图函数 (新)
# ==========================================

def plot_noise_sensitivity(multipliers, r2_scores, accuracies, output_path):
    """
    绘制噪声敏感性分析图
    
    Args:
        multipliers: 噪声倍率列表 (x轴)
        r2_scores: 对应的 R2 分数列表 (左y轴)
        accuracies: 对应的准确率列表 (右y轴)
        output_path: 保存路径
    """
    fig, ax1 = plt.subplots(figsize=(8, 6))

    # 设置颜色
    color_r2 = '#D62728'  # 砖红色
    color_acc = '#1F77B4' # 经典蓝

    # --- 绘制左侧 Y轴 (R2) ---
    ax1.set_xlabel(r'Noise Factor Multiplier ($n \times 0.01$)')
    ax1.set_ylabel(r'Average $R^2$ Score', color=color_r2)
    
    # 绘制带标记的线
    line1 = ax1.plot(multipliers, r2_scores, marker='o', linestyle='-', 
                     linewidth=2, markersize=8, color=color_r2, label=r'Avg. $R^2$')
    
    ax1.tick_params(axis='y', labelcolor=color_r2, direction='in')
    
    # 设置 R2 的范围 (根据你的描述接近0.9，留出一点空间)
    # ax1.set_ylim(0.0, 1.0) # 如果需要固定范围取消注释

    # --- 绘制右侧 Y轴 (Accuracy) ---
    ax2 = ax1.twinx()  # 实例化第二个轴
    ax2.set_ylabel('Accuracy (%)', color=color_acc)
    
    line2 = ax2.plot(multipliers, accuracies, marker='s', linestyle='--', 
                     linewidth=2, markersize=8, color=color_acc, label='Accuracy')
    
    ax2.tick_params(axis='y', labelcolor=color_acc, direction='in', right=True)
    
    # 设置 Acc 的范围
    # ax2.set_ylim(0, 100) # 如果需要固定范围取消注释

    # --- 合并图例 ---
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    # 将图例放在合适的位置，通常是中间偏右或根据数据调整
    ax1.legend(lines, labels, loc='center right', frameon=True, edgecolor='gray', framealpha=0.9)

    # 添加网格 (可选)
    ax1.grid(True, linestyle=':', alpha=0.6)

    plt.title("Model Robustness to Spatial Gaussian Noise", y=1.02, fontsize=16)
    plt.tight_layout()
    
    # 保存
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 敏感性分析图已保存至: {output_path}")
    plt.close()

# ==========================================
# 主程序
# ==========================================

def main():
    # 1. 加载配置
    config_path = 'config_test_inch_plot.yaml'
    if not os.path.exists(config_path):
        print(f"Error: Config file {config_path} not found.")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 强制关闭不需要的功能
    config['interpretability']['enabled'] = False
    config['test_config']['save_features']['enabled'] = False
    
    # 2. 初始化环境
    set_seed(config['seed'])
    device = get_device(config['device'])
    print(f"使用设备: {device}")
    
    output_dir = Path(config['output_dir']) / config['experiment_name'] / 'sensitivity_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 3. 加载模型 (只需加载一次，因为权重不变)
    checkpoint_path = config['test_config']['model_checkpoint']
    model = load_model_from_checkpoint(checkpoint_path, config, device)
    
    # 4. 定义敏感性分析的参数
    # n 的倍数：从 0 (无噪声) 到 5 倍或 10 倍标准噪声
    # 这里你可以根据需要修改 range
    noise_multipliers = [ 0.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0] 
    base_factor = 0.01 # 这是标准值
    
    history_r2 = []
    history_acc = []
    
    print("\n" + "="*60)
    print("开始噪声敏感性分析 (R2 & Accuracy)")
    print("="*60)
    
    # 5. 循环测试
    for n in noise_multipliers:
        current_factor = base_factor * n
        print(f"\n>>> Testing with Noise Factor: {current_factor:.4f} (Multiplier: {n}x)")
        
        # --- 关键：动态修改 Config 并重新创建 Dataset ---
        # 必须重新创建 Dataset，因为噪声是在 __getitem__ 中根据 config 应用的
        config['preprocessing']['gaussian_spatial_noise']['factor'] = current_factor
        
        # 获取测试数据路径
        test_data_path = config['test_config'].get('test_data_path', '')
        if not test_data_path:
            test_data_path = Path(config['data_path']) / 'test'
        
        # 重新初始化数据集和加载器
        test_dataset = TestMultiTaskDataset(
            test_data_path, 
            config=config, 
            split='test' 
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=config['test_config']['save_features']['batch_size'], 
            shuffle=False, 
            num_workers=config['num_workers']
        )
        
        # 执行测试
        results = test_model(model, test_loader, device)
        metrics = evaluate_results(results)
        
        # 记录结果
        avg_r2 = metrics['avg_r2'] # 或者使用 metrics['r2_beta2'] 单独分析
        acc = metrics['classification_accuracy']
        
        history_r2.append(avg_r2)
        history_acc.append(acc)
        
        print(f"   -> Avg R2: {avg_r2:.4f}")
        print(f"   -> Accuracy: {acc:.2f}%")

    # 6. 绘制结果
    print("\n" + "="*60)
    print("正在生成分析图表...")
    
    plot_path = output_dir / 'noise_sensitivity_R2_Acc.pdf' # 同时保存 PDF 和 PNG
    plot_noise_sensitivity(noise_multipliers, history_r2, history_acc, plot_path)
    
    plot_path_png = output_dir / 'noise_sensitivity_R2_Acc.png'
    plot_noise_sensitivity(noise_multipliers, history_r2, history_acc, plot_path_png)
    
    # 保存原始数据以备后用
    data_path = output_dir / 'sensitivity_data.npz'
    np.savez(data_path, multipliers=noise_multipliers, r2=history_r2, acc=history_acc)
    print(f"数据已保存至: {data_path}")
    print("="*60)

if __name__ == '__main__':
    main()