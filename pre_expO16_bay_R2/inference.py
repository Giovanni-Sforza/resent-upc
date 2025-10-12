import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import yaml
import argparse
from pathlib import Path
import re
from train import PreprocessTransform, create_model # 从训练脚本中复用模块

def predict_beta(model, image_path, sorted_betas, device):
    """
    对单个图像执行贝叶斯推断
    
    Args:
        model (nn.Module): 已加载的、训练好的模型
        image_path (str or Path): 输入的.npy图像文件路径
        sorted_betas (list): 训练时使用的beta值列表
        device (torch.device): 运行推断的设备
        
    Returns:
        tuple: (mean_beta, std_dev_beta, probabilities)
    """
    # 1. 加载和预处理图像
    try:
        image = np.load(image_path)
    except Exception as e:
        print(f"错误: 无法加载图像文件 {image_path}. {e}")
        return None, None, None

    # 使用与训练时完全相同的变换
    transform = PreprocessTransform()
    image_tensor = torch.from_numpy(image).float().unsqueeze(0) # (1, H, W)
    input_tensor = transform(image_tensor).unsqueeze(0).to(device) # (1, 3, 224, 224)
    
    # 2. 模型前向传播
    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        # 3. 应用Softmax获取概率
        probabilities = torch.softmax(logits, dim=1).cpu().numpy().flatten()

    # 4. 计算贝叶斯推断结果
    betas_np = np.array(sorted_betas)
    
    # 计算均值 (点估计)
    mean_beta = np.sum(probabilities * betas_np)
    
    # 计算方差和标准差 (不确定度)
    variance_beta = np.sum(probabilities * ((betas_np - mean_beta)**2))
    std_dev_beta = np.sqrt(variance_beta)
    
    return mean_beta, std_dev_beta, probabilities

def main():
    parser = argparse.ArgumentParser(description="Bayesian Inference for Nuclear Deformation Parameters")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model checkpoint (.pth file)")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input .npy image file for inference")
    parser.add_argument('--param_name', type=str, default='beta', help="Name of the parameter for display (e.g., 'beta2' or 'beta3')")
    parser.add_argument('--device', type=str, default='auto', choices=['cuda', 'cpu', 'auto'], help="Device to use for inference")
    
    args = parser.parse_args()

    # 确定设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # 加载模型检查点
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"错误: 模型文件未找到 at {model_path}")
        return
        
    checkpoint = torch.load(model_path, map_location=device)
    
    # 从检查点中提取关键信息
    if 'sorted_betas' not in checkpoint:
        print("错误: 模型检查点中未找到 'sorted_betas' 列表。请使用新的训练脚本重新训练。")
        return
        
    sorted_betas = checkpoint['sorted_betas']
    num_classes = len(sorted_betas)
    
    # 创建模型实例并加载权重
    model = create_model(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # 执行推断
    mean_beta, std_dev_beta, probabilities = predict_beta(
        model=model,
        image_path=args.image_path,
        sorted_betas=sorted_betas,
        device=device
    )
    
    if mean_beta is None:
        return # 如果图像加载失败则退出
        
    # 打印结果
    print("\n" + "="*50)
    print(f"--- Bayesian Inference Result for {args.param_name} ---")
    print(f"Point Estimate (Mean) : {mean_beta:.6f}")
    print(f"Uncertainty (Std Dev) : {std_dev_beta:.6f}")
    print("="*50)
    
    print("\nProbability Distribution over known betas:")
    for beta, prob in zip(sorted_betas, probabilities):
        print(f"  - P({args.param_name} = {beta:.4f}) = {prob:.2%}")
        
if __name__ == '__main__':
    main()