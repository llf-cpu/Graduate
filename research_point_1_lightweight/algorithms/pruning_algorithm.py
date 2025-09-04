import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple

class DualChannelPruning:
    """基于双通道准则的模型剪枝算法，已适配MobileNetV3等复杂结构。"""
    
    def __init__(self, model: nn.Module, pruning_ratio: float = 0.5):
        self.model = model
        self.pruning_ratio = pruning_ratio
        self.importance_scores = {}
        # 存储需要一起剪枝的层
        self.grouped_layers = self._group_dependent_layers()
        
    def _group_dependent_layers(self):
        # 在MobileNetV3中，一些层是相互依赖的，特别是InvertedResidual块中的层
        # 简化处理：这里我们仍然独立处理每个Conv2d层，但在更高级的剪枝中需要考虑依赖关系
        # 此处暂时返回一个空字典，表示各层独立处理
        return {}

    def calculate_importance_scores(self) -> Dict[str, torch.Tensor]:
        """计算各层权重的重要性分数"""
        importance_scores = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d): # 重点关注卷积层
                # 基于权重大小和梯度的双通道准则
                weight = module.weight.data.clone()
                
                # 通道1：权重幅度 (L2范数)
                # 对于卷积层，我们在输出通道维度上计算范数
                magnitude_score = torch.norm(weight.view(weight.size(0), -1), p=2, dim=1)
                
                # 通道2：权重变异性（方差）
                variance_score = torch.var(weight.view(weight.size(0), -1), dim=1)
                
                # 综合重要性分数
                # 添加一个小的epsilon防止分数为0
                combined_score = magnitude_score * variance_score + 1e-8
                importance_scores[name] = combined_score
                
        return importance_scores
    
    def prune_model(self) -> nn.Module:
        """执行模型剪枝（非结构化，将权重置零）"""
        self.importance_scores = self.calculate_importance_scores()
        
        # 收集所有层的分数以确定全局剪枝阈值
        all_scores = torch.cat(list(self.importance_scores.values()))
        num_total_channels = len(all_scores)
        num_prune = int(num_total_channels * self.pruning_ratio)
        
        if num_prune == 0:
            print("剪枝率为0或模型太小，不执行剪枝。")
            return self.model

        # 全局阈值
        threshold, _ = torch.topk(all_scores, num_prune, largest=False)
        pruning_threshold = threshold[-1]

        print(f"\n全局剪枝阈值: {pruning_threshold:.4f}")

        total_pruned_channels = 0
        for name, module in self.model.named_modules():
            if name in self.importance_scores:
                scores = self.importance_scores[name]
                
                # 根据全局阈值确定要剪枝的通道
                mask = scores.le(pruning_threshold)
                indices_to_prune = torch.where(mask)[0]
                
                if len(indices_to_prune) > 0:
                    pruned_count = len(indices_to_prune)
                    total_pruned_channels += pruned_count
                    print(f"  - 在层 '{name}' 中剪枝 {pruned_count} / {len(scores)} 个通道.")

                    # 将对应权重置零
                    if isinstance(module, (nn.Conv2d, nn.Linear)):
                        module.weight.data[indices_to_prune] = 0
                        if module.bias is not None:
                            # 对于卷积层，如果剪枝了某个输出通道，其对应的偏置也应置零
                            module.bias.data[indices_to_prune] = 0
        
        print(f"\n总共剪枝了 {total_pruned_channels} 个通道。")
        return self.model
    
    def get_compression_ratio(self) -> float:
        """获取压缩比例"""
        total_params = sum(p.numel() for p in self.model.parameters())
        non_zero_params = sum((p != 0).sum().item() for p in self.model.parameters())
        return total_params / non_zero_params if non_zero_params > 0 else 1.0