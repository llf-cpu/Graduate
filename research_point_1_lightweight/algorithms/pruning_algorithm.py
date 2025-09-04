import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple

class DualChannelPruning:
    """基于双通道准则的模型剪枝算法"""
    
    def __init__(self, model: nn.Module, pruning_ratio: float = 0.5):
        self.model = model
        self.pruning_ratio = pruning_ratio
        self.importance_scores = {}
        
    def calculate_importance_scores(self) -> Dict[str, torch.Tensor]:
        """计算各层权重的重要性分数"""
        importance_scores = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # 基于权重大小和梯度的双通道准则
                weight = module.weight.data
                
                # 通道1：权重幅度
                magnitude_score = torch.norm(weight.view(weight.size(0), -1), dim=1)
                
                # 通道2：权重变异性
                variance_score = torch.var(weight.view(weight.size(0), -1), dim=1)
                
                # 综合重要性分数
                combined_score = magnitude_score * variance_score
                importance_scores[name] = combined_score
                
        return importance_scores
    
    def prune_model(self) -> nn.Module:
        """执行模型剪枝"""
        self.importance_scores = self.calculate_importance_scores()
        
        for name, module in self.model.named_modules():
            if name in self.importance_scores:
                scores = self.importance_scores[name]
                num_channels = len(scores)
                num_prune = int(num_channels * self.pruning_ratio)
                
                # 选择重要性分数最低的通道进行剪枝
                _, indices = torch.topk(scores, num_prune, largest=False)
                
                # 将对应权重置零
                if isinstance(module, nn.Conv2d):
                    module.weight.data[indices] = 0
                elif isinstance(module, nn.Linear):
                    module.weight.data[indices] = 0
                    
        return self.model
    
    def get_compression_ratio(self) -> float:
        """获取压缩比例"""
        total_params = sum(p.numel() for p in self.model.parameters())
        non_zero_params = sum((p != 0).sum().item() for p in self.model.parameters())
        return total_params / non_zero_params if non_zero_params > 0 else 1.0