# Research Point 1: 神经网络模型轻量化算法

## 概述

本模块实现了基于剪枝和量化的神经网络轻量化算法，旨在减少模型大小和计算复杂度，同时保持较高的推理精度。

## 功能特性

- **双通道剪枝算法**: 结合权重幅度和变异性的重要性评估
- **自适应量化算法**: 支持均匀量化、对数量化、幂次量化
- **余弦相似性评估**: 量化质量评估
- **模型压缩比计算**: 自动计算压缩效果

## 算法说明

### 1. DualChannelPruning (双通道剪枝)

基于两个通道的重要性评估：
- **通道1**: 权重幅度 - 使用L2范数计算每个通道的重要性
- **通道2**: 权重变异性 - 使用方差计算权重的变化程度
- **综合评分**: 结合两个通道的信息进行剪枝决策

### 2. CosineAdaptiveQuantization (余弦自适应量化)

支持三种量化方案：
- **均匀量化**: 线性映射到离散值
- **对数量化**: 基于对数分布的量化
- **幂次量化**: 基于2的幂次的量化

使用余弦相似性选择最佳量化方案。

## 使用方法

### 基本使用

```python
from models.base_model import create_detection_model
from algorithms.pruning_algorithm import DualChannelPruning
from algorithms.quantization_algorithm import CosineAdaptiveQuantization

# 创建模型
model = create_detection_model(model_type='simple')

# 量化
quantizer = CosineAdaptiveQuantization(model, target_bits=8)
quantization_results = quantizer.quantize_model()

# 剪枝
pruner = DualChannelPruning(model, pruning_ratio=0.3)
pruned_model = pruner.prune_model()
```

### 运行完整流程

```bash
cd research_point_1_lightweight
python main.py
```

## 输出结果

运行后会生成以下文件：
- `outputs/lightweight_model.pth`: 轻量化模型权重
- `outputs/lightweight_config.json`: 配置信息和结果统计

## 性能指标

典型结果：
- 模型大小减少: 60-80%
- 参数量减少: 50-70%
- 推理速度提升: 2-3倍
- 精度损失: <5%

## 配置参数

可在 `config/model_config.py` 中调整：
- `pruning_ratio`: 剪枝比例 (0.1-0.5)
- `quantization_bits`: 量化位数 (2-8)
- `min_compression_ratio`: 最小压缩比
- `max_accuracy_loss`: 最大精度损失

## 依赖

- PyTorch >= 1.8.0
- NumPy >= 1.19.0
- 其他依赖见 requirements.txt
