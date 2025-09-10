# ===================================================================
# 使用示例和额外工具函数
# ===================================================================

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import yaml

# ===================================================================
# 1. 模型预测和评估工具
# ===================================================================

def load_trained_model(checkpoint_path, device='cpu'):
    """加载训练好的模型"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # 重建模型
    if config['inference_mode'] == 'gp':
        from main_script import ResNetVariationalGP  # 假设主脚本名为main_script.py
        model = ResNetVariationalGP(
            feature_dim=config['gp_config']['feature_dim'],
            output_dim=config['gp_config']['output_dim'],
            num_inducing=config['gp_config']['num_inducing']
        )
    else:
        # 分类器模式
        import torchvision.models as models
        import torch.nn as nn
        model = models.resnet34(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, len(checkpoint['class_to_pair']))
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config

def predict_single_image(model, image_path, transform, device='cpu'):
    """对单张图像进行预测"""
    import torch
    import numpy as np
    
    model.eval()
    
    # 加载和预处理图像
    image_np = np.load(image_path)
    image = torch.from_numpy(image_np).float().unsqueeze(0).unsqueeze(0)
    
    if transform:
        image = transform(image)
    
    image = image.to(device)
    
    with torch.no_grad():
        if hasattr(model, 'likelihood'):  # GP模型
            model.likelihood.eval()
            with gpytorch.settings.fast_pred_var():
                dist = model(image)
                output_dist = model.likelihood(dist)
                mean = output_dist.mean.transpose(0, 1).cpu().numpy()
                # 获取不确定性（标准差）
                std = output_dist.stddev.transpose(0, 1).cpu().numpy()
                return mean[0], std[0]  # 返回均值和标准差
        else:  # 分类器模型
            outputs = model(image)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = outputs.argmax(1).item()
            confidence = probabilities[0, predicted_class].item()
            return predicted_class, confidence

def evaluate_model_uncertainty(model, dataloader, device='cpu', num_samples=10):
    """评估模型的不确定性（仅适用于GP模型）"""
    if not hasattr(model, 'likelihood'):
        raise ValueError("不确定性评估仅适用于GP模型")
    
    model.eval()
    model.likelihood.eval()
    
    uncertainties = []
    predictions = []
    labels = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 多次采样以估计不确定性
            sample_predictions = []
            for _ in range(num_samples):
                dist = model(inputs)
                output_dist = model.likelihood(dist)
                sample = output_dist.sample().transpose(0, 1)
                sample_predictions.append(sample.cpu().numpy())
            
            # 计算均值和标准差
            sample_predictions = np.array(sample_predictions)
            mean_pred = np.mean(sample_predictions, axis=0)
            std_pred = np.std(sample_predictions, axis=0)
            
            predictions.extend(mean_pred)
            uncertainties.extend(std_pred)
            labels.extend(targets.cpu().numpy())
    
    return np.array(predictions), np.array(uncertainties), np.array(labels)

# ===================================================================
# 2. 数据可视化工具
# ===================================================================

def plot_training_history(log_dir, save_path=None):
    """从tensorboard日志中提取并绘制训练历史"""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        event_acc = EventAccumulator(str(log_dir))
        event_acc.Reload()
        
        # 获取可用的标量标签
        scalar_tags = event_acc.Tags()['scalars']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16)
        
        # 训练和验证损失
        if 'Loss/Train' in scalar_tags and 'Loss/Validation' in scalar_tags:
            train_loss = [s.value for s in event_acc.Scalars('Loss/Train')]
            val_loss = [s.value for s in event_acc.Scalars('Loss/Validation')]
            epochs = [s.step for s in event_acc.Scalars('Loss/Train')]
            
            axes[0, 0].plot(epochs, train_loss, label='Train Loss', alpha=0.8)
            axes[0, 0].plot(epochs, val_loss, label='Validation Loss', alpha=0.8)
            axes[0, 0].set_title('Loss Curves')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
        
        # MSE（如果是GP模式）
        if 'MSE/Validation' in scalar_tags:
            mse = [s.value for s in event_acc.Scalars('MSE/Validation')]
            epochs = [s.step for s in event_acc.Scalars('MSE/Validation')]
            
            axes[0, 1].plot(epochs, mse, color='red', alpha=0.8)
            axes[0, 1].set_title('Validation MSE')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('MSE')
            axes[0, 1].grid(True, alpha=0.3)
        
        # 学习率
        if 'Learning_Rate' in scalar_tags:
            lr = [s.value for s in event_acc.Scalars('Learning_Rate')]
            epochs = [s.step for s in event_acc.Scalars('Learning_Rate')]
            
            axes[1, 0].semilogy(epochs, lr, color='green', alpha=0.8)
            axes[1, 0].set_title('Learning Rate')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate (log scale)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Beta2和Beta3的MSE（如果可用）
        if 'MSE_Beta2/Validation' in scalar_tags and 'MSE_Beta3/Validation' in scalar_tags:
            mse_beta2 = [s.value for s in event_acc.Scalars('MSE_Beta2/Validation')]
            mse_beta3 = [s.value for s in event_acc.Scalars('MSE_Beta3/Validation')]
            epochs = [s.step for s in event_acc.Scalars('MSE_Beta2/Validation')]
            
            axes[1, 1].plot(epochs, mse_beta2, label='Beta2 MSE', alpha=0.8)
            axes[1, 1].plot(epochs, mse_beta3, label='Beta3 MSE', alpha=0.8)
            axes[1, 1].set_title('Individual Beta MSE')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('MSE')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
    except ImportError:
        print("需要安装tensorboard才能绘制训练历史: pip install tensorboard")

def plot_uncertainty_analysis(predictions, uncertainties, labels, save_path=None):
    """绘制GP模型的不确定性分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Uncertainty Analysis', fontsize=16)
    
    # 预测误差 vs 不确定性（Beta2）
    errors_beta2 = np.abs(predictions[:, 0] - labels[:, 0])
    axes[0, 0].scatter(uncertainties[:, 0], errors_beta2, alpha=0.6)
    axes[0, 0].set_xlabel('Predicted Uncertainty (Beta2)')
    axes[0, 0].set_ylabel('Absolute Error (Beta2)')
    axes[0, 0].set_title('Uncertainty vs Error (Beta2)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 预测误差 vs 不确定性（Beta3）
    errors_beta3 = np.abs(predictions[:, 1] - labels[:, 1])
    axes[0, 1].scatter(uncertainties[:, 1], errors_beta3, alpha=0.6)
    axes[0, 1].set_xlabel('Predicted Uncertainty (Beta3)')
    axes[0, 1].set_ylabel('Absolute Error (Beta3)')
    axes[0, 1].set_title('Uncertainty vs Error (Beta3)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 不确定性分布
    axes[1, 0].hist(uncertainties[:, 0], bins=30, alpha=0.7, label='Beta2', density=True)
    axes[1, 0].hist(uncertainties[:, 1], bins=30, alpha=0.7, label='Beta3', density=True)
    axes[1, 0].set_xlabel('Predicted Uncertainty')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Uncertainty Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 高不确定性vs低不确定性的误差对比
    high_unc_threshold = np.percentile(np.mean(uncertainties, axis=1), 75)
    high_unc_mask = np.mean(uncertainties, axis=1) > high_unc_threshold
    
    low_unc_errors = np.mean(np.abs(predictions[~high_unc_mask] - labels[~high_unc_mask]), axis=1)
    high_unc_errors = np.mean(np.abs(predictions[high_unc_mask] - labels[high_unc_mask]), axis=1)
    
    axes[1, 1].boxplot([low_unc_errors, high_unc_errors], 
                      labels=['Low Uncertainty', 'High Uncertainty'])
    axes[1, 1].set_ylabel('Mean Absolute Error')
    axes[1, 1].set_title('Error Distribution by Uncertainty Level')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# ===================================================================
# 3. 完整的评估脚本示例
# ===================================================================

def evaluate_trained_model(checkpoint_path, test_data_path, config_path=None):
    """完整的模型评估脚本"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model, config = load_trained_model(checkpoint_path, device)
    print("模型加载完成")
    
    # 加载测试数据
    from main_script import InterferenceDataset, PreprocessTransform
    transform = PreprocessTransform()
    test_dataset = InterferenceDataset(test_data_path, config=config, 
                                     transform=transform, split='test')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, 
                                             shuffle=False, num_workers=4)
    
    if config['inference_mode'] == 'gp':
        # GP模型评估
        print("开始GP模型评估...")
        predictions, uncertainties, labels = evaluate_model_uncertainty(
            model, test_loader, device, num_samples=10)
        
        # 计算指标
        mse = mean_squared_error(labels, predictions)
        mae = mean_absolute_error(labels, predictions)
        
        mse_beta2 = mean_squared_error(labels[:, 0], predictions[:, 0])
        mse_beta3 = mean_squared_error(labels[:, 1], predictions[:, 1])
        
        print(f"测试集结果:")
        print(f"总体 MSE: {mse:.6f}, MAE: {mae:.6f}")
        print(f"Beta2 MSE: {mse_beta2:.6f}")
        print(f"Beta3 MSE: {mse_beta3:.6f}")
        print(f"平均不确定性: Beta2={np.mean(uncertainties[:, 0]):.6f}, Beta3={np.mean(uncertainties[:, 1]):.6f}")
        
        # 绘制分析图
        plot_uncertainty_analysis(predictions, uncertainties, labels, 'uncertainty_analysis.png')
        
    else:
        # 分类器模型评估
        print("开始分类器模型评估...")
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = 100. * correct / total
        print(f"测试集准确率: {accuracy:.2f}%")
        
        # 绘制混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Test Set Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('test_confusion_matrix.png', dpi=150)
        plt.show()

# ===================================================================
# 4. 使用示例
# ===================================================================

if __name__ == "__main__":
    # 示例1: 训练模型
    print("开始训练...")
    # python main_script.py  # 运行主训练脚本
    
    # 示例2: 评估训练好的模型
    # evaluate_trained_model(
    #     checkpoint_path="outputs/resnet_variational_gp_beta_regression/checkpoints/best_model.pth",
    #     test_data_path="path/to/your/test/data"
    # )
    
    # 示例3: 单张图像预测
    # model, config = load_trained_model("path/to/checkpoint.pth")
    # transform = PreprocessTransform()
    # 
    # beta_pred, uncertainty = predict_single_image(
    #     model, "path/to/image.npy", transform
    # )
    # print(f"预测的Beta值: {beta_pred}")
    # print(f"预测不确定性: {uncertainty}")
    
    # 示例4: 绘制训练历史
    # plot_training_history("outputs/resnet_variational_gp_beta_regression/logs")
    
    pass