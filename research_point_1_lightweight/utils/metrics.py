import torch
import torch.nn as nn
from typing import Dict

def calculate_model_metrics(model: nn.Module) -> Dict[str, float]:
    """计算模型指标"""
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 计算模型大小（MB）
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(buf.numel() * buf.element_size() for buf in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 * 1024)
    
    # 简化的FLOPs估算
    flops = estimate_flops(model)
    
    return {
        'params': total_params,
        'trainable_params': trainable_params,
        'size_mb': model_size_mb,
        'flops': flops,
        'compression_ratio': 1.0  # 原始模型的压缩比为1
    }

def estimate_flops(model: nn.Module, input_size: tuple = (1, 3, 224, 224)) -> float:
    """估算模型FLOPs"""
    total_flops = 0
    
    def conv_flops(module, input_shape, output_shape):
        kernel_flops = module.kernel_size[0] * module.kernel_size[1]
        output_elements = output_shape[2] * output_shape[3]
        flops = kernel_flops * module.in_channels * output_elements * module.out_channels
        return flops
    
    def linear_flops(module, input_shape, output_shape):
        return module.in_features * module.out_features
    
    # 简化的FLOPs计算
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # 假设输出尺寸为输入的1/4（由于pooling等操作）
            estimated_flops = module.kernel_size[0] * module.kernel_size[1] * \
                            module.in_channels * module.out_channels * (224 // 4) * (224 // 4)
            total_flops += estimated_flops
        elif isinstance(module, nn.Linear):
            total_flops += module.in_features * module.out_features
    
    return total_flops

def compare_models(original_model: nn.Module, lightweight_model: nn.Module) -> Dict[str, float]:
    """比较原始模型和轻量化模型的指标"""
    
    # 计算两个模型的指标
    original_metrics = calculate_model_metrics(original_model)
    lightweight_metrics = calculate_model_metrics(lightweight_model)
    
    # 计算压缩比
    compression_ratio = original_metrics['params'] / lightweight_metrics['params']
    
    # 计算减少百分比
    size_reduction = (1 - lightweight_metrics['size_mb'] / original_metrics['size_mb']) * 100
    flops_reduction = (1 - lightweight_metrics['flops'] / original_metrics['flops']) * 100
    
    return {
        'compression_ratio': compression_ratio,
        'size_reduction': size_reduction,
        'flops_reduction': flops_reduction,
        'original_metrics': original_metrics,
        'lightweight_metrics': lightweight_metrics
    }