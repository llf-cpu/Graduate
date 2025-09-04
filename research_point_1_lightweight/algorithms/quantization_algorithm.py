import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
import math

class CosineAdaptiveQuantization:
    """
    基于余弦相似性的自适应量化算法。
    为模型的每一层选择最佳的量化策略（均匀、对数或二的幂）。
    """
    
    def __init__(self, model: nn.Module, target_bits: int = 8):
        self.model = model
        self.target_bits = target_bits
        self.quantization_schemes = ['uniform', 'logarithmic', 'power_of_two']
        print(f"初始化自适应量化器，目标位宽: {target_bits}-bit")
        
    def calculate_cosine_similarity(self, original: torch.Tensor, 
                                  quantized: torch.Tensor) -> float:
        """计算原始权重与量化权重的余弦相似性"""
        original_flat = original.flatten().to(torch.float32)
        quantized_flat = quantized.flatten().to(torch.float32)
        
        # 避免零向量导致NaN
        if torch.norm(original_flat) == 0 or torch.norm(quantized_flat) == 0:
            return 1.0 if torch.norm(original_flat) == torch.norm(quantized_flat) else 0.0

        cosine_sim = torch.cosine_similarity(
            original_flat.unsqueeze(0), 
            quantized_flat.unsqueeze(0)
        ).item()
        
        return cosine_sim
    
    def uniform_quantization(self, tensor: torch.Tensor, bits: int) -> torch.Tensor:
        """均匀量化"""
        max_val = tensor.abs().max()
        if max_val == 0: return tensor
        scale = max_val / (2**(bits-1) - 1)
        if scale == 0: return tensor
        quantized = torch.round(tensor / scale) * scale
        return torch.clamp(quantized, -max_val, max_val)
    
    def logarithmic_quantization(self, tensor: torch.Tensor, bits: int) -> torch.Tensor:
        """对数量化"""
        sign = torch.sign(tensor)
        abs_tensor = torch.abs(tensor)
        
        # 避免log(0)
        abs_tensor = torch.clamp(abs_tensor, min=1e-9)
        
        log_tensor = torch.log2(abs_tensor)
        max_log = log_tensor.max()
        min_log = log_tensor.min()

        # 确定缩放和零点
        scale = (max_log - min_log) / (2**bits - 1)
        if scale == 0: return tensor # 如果张量中所有值相同
        
        zero_point = torch.round(min_log / scale)
        
        # 量化和反量化
        quantized_log = (torch.round(log_tensor / scale) - zero_point) * scale + min_log
        
        quantized = sign * torch.pow(2, quantized_log)
        return quantized
    
    def power_of_two_quantization(self, tensor: torch.Tensor, bits: int) -> torch.Tensor:
        """2的幂次量化"""
        sign = torch.sign(tensor)
        abs_tensor = torch.abs(tensor)
        
        # 找到最接近的2的幂次
        log2_tensor = torch.log2(torch.clamp(abs_tensor, min=1e-9))
        rounded_log2 = torch.round(log2_tensor)
        
        # 这是一个简化的实现
        max_exp = 2**(bits-1) - 1
        rounded_log2 = torch.clamp(rounded_log2, -max_exp, max_exp)
        
        quantized = sign * torch.pow(2, rounded_log2)
        return quantized
    
    def select_best_quantization(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, str, float]:
        """为给定的张量选择最佳量化方案"""
        best_similarity = -2.0
        best_quantized = tensor.clone()
        best_scheme = 'none'
        
        if torch.all(tensor == 0):
            return best_quantized, 'none', 1.0

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
                print(f"  - 警告: 量化方案 {scheme_name} 失败: {e}")
                continue
        
        if best_scheme == 'none':
            print("  - 警告: 所有量化方案均失败，该层未被量化。")

        return best_quantized, best_scheme, best_similarity
    
    def quantize_model(self) -> Dict[str, Any]:
        """量化整个模型"""
        quantization_results = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and module.weight is not None:
                print(f"- 正在量化层: {name}")
                original_weight = module.weight.data.clone()
                
                quantized_weight, best_scheme, similarity = self.select_best_quantization(original_weight)
                
                module.weight.data = quantized_weight
                
                quantization_results[name] = {
                    'scheme': best_scheme,
                    'similarity': similarity
                }
                print(f"  - 最佳方案: {best_scheme}, 余弦相似性: {similarity:.4f}")

        print("\n模型量化完成。")
        return quantization_results
