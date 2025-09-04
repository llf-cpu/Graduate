import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
import math

class CosineAdaptiveQuantization:
    """基于余弦相似性的自适应量化算法"""
    
    def __init__(self, model: nn.Module, target_bits: int = 8):
        self.model = model
        self.target_bits = target_bits
        self.quantization_schemes = ['uniform', 'logarithmic', 'power_of_two']
        
    def calculate_cosine_similarity(self, original: torch.Tensor, 
                                  quantized: torch.Tensor) -> float:
        """计算原始权重与量化权重的余弦相似性"""
        original_flat = original.flatten()
        quantized_flat = quantized.flatten()
        
        cosine_sim = torch.cosine_similarity(
            original_flat.unsqueeze(0), 
            quantized_flat.unsqueeze(0)
        ).item()
        
        return cosine_sim
    
    def uniform_quantization(self, tensor: torch.Tensor, bits: int) -> torch.Tensor:
        """均匀量化"""
        scale = tensor.abs().max() / (2**(bits-1) - 1)
        quantized = torch.round(tensor / scale) * scale
        return torch.clamp(quantized, -tensor.abs().max(), tensor.abs().max())
    
    def logarithmic_quantization(self, tensor: torch.Tensor, bits: int) -> torch.Tensor:
        """对数量化"""
        sign = torch.sign(tensor)
        abs_tensor = torch.abs(tensor)
        
        # 避免log(0)
        abs_tensor = torch.clamp(abs_tensor, min=1e-8)
        
        log_tensor = torch.log2(abs_tensor)
        scale = log_tensor.max() / (2**(bits-1) - 1)
        quantized_log = torch.round(log_tensor / scale) * scale
        
        quantized = sign * torch.pow(2, quantized_log)
        return quantized
    
    def power_of_two_quantization(self, tensor: torch.Tensor, bits: int) -> torch.Tensor:
        """2的幂次量化"""
        sign = torch.sign(tensor)
        abs_tensor = torch.abs(tensor)
        
        # 找到最接近的2的幂次
        log2_tensor = torch.log2(torch.clamp(abs_tensor, min=1e-8))
        rounded_log2 = torch.round(log2_tensor)
        
        # 限制在bits范围内
        max_exp = 2**(bits-1) - 1
        rounded_log2 = torch.clamp(rounded_log2, -max_exp, max_exp)
        
        quantized = sign * torch.pow(2, rounded_log2)
        return quantized
    
    def select_best_quantization(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, str]:
        """选择最佳量化方案"""
        best_similarity = -1
        best_quantized = None
        best_scheme = None
        
        schemes = {
            'uniform': self.uniform_quantization,
            'logarithmic': self.logarithmic_quantization,
            'power_of_two': self.power_of_two_quantization
        }
        
        for scheme_name, quantize_func in schemes.items():
            try:
                quantized = quantize_func(tensor, self.target_bits)
                similarity = self.calculate_cosine_similarity(tensor, quantized)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_quantized = quantized
                    best_scheme = scheme_name
            except Exception as e:
                print(f"量化方案 {scheme_name} 失败: {e}")
                continue
                
        return best_quantized, best_scheme
    
    def quantize_model(self) -> Dict[str, Any]:
        """量化整个模型"""
        quantization_results = {}
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                original_weight = module.weight.data.clone()
                quantized_weight, best_scheme = self.select_best_quantization(original_weight)
                
                # 更新模型权重
                module.weight.data = quantized_weight
                
                quantization_results[name] = {
                    'scheme': best_scheme,
                    'similarity': self.calculate_cosine_similarity(original_weight, quantized_weight),
                    'compression_ratio': self.target_bits / 32  # 假设原始为32位
                }
                
        return quantization_results
