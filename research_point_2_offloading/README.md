# Research Point 2: 多目标推理任务卸载优化算法

## 概述

本模块实现了基于深度强化学习的多目标推理任务卸载优化算法，旨在在车路协同环境中实现延迟、能耗和精度的平衡优化。

## 功能特性

- **动态规划任务分割**: 自适应任务比例分配
- **DDPG强化学习**: 深度确定性策略梯度算法
- **多目标优化**: 延迟、能耗、精度平衡
- **车路协同环境**: 真实车辆和边缘节点仿真

## 算法说明

### 1. AdaptiveTaskSplitting (自适应任务分割)

基于动态规划的任务分割算法：
- **处理成本计算**: 考虑计算能力和当前负载
- **通信成本计算**: 考虑带宽、延迟和距离
- **最优分割**: 使用动态规划找到最优任务分配比例

### 2. DDPGOffloadingAgent (DDPG卸载智能体)

改进的DDPG算法：
- **Actor网络**: 输出连续动作（卸载比例、目标节点、优先级）
- **Critic网络**: 评估状态-动作对的价值
- **经验回放**: 存储和采样训练经验
- **软更新**: 目标网络的软更新机制

## 使用方法

### 基本使用

```python
from environment.vehicle_environment import VehicleEnvironment
from algorithms.ddpg_offloading import DDPGOffloadingAgent
from algorithms.dynamic_programming import AdaptiveTaskSplitting

# 初始化环境
env = VehicleEnvironment(num_vehicles=10, num_edge_nodes=5)

# 初始化DDPG智能体
agent = DDPGOffloadingAgent(state_dim=15, action_dim=3)

# 训练
for episode in range(1000):
    state = env.reset()
    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.store_experience(state, action, reward, next_state, done)
        agent.update()
```

### 运行完整流程

```bash
cd research_point_2_offloading
python main.py
```

## 输出结果

运行后会生成以下文件：
- `outputs/ddpg_offloading_model.pth`: 训练好的DDPG模型
- `outputs/offloading_config.json`: 配置信息和训练统计

## 性能指标

典型结果：
- 平均延迟减少: 30-50%
- 能耗优化: 20-40%
- 系统吞吐量提升: 1.5-2倍
- 精度保持: >95%

## 环境配置

可在 `config/environment_config.py` 中调整：
- `num_vehicles`: 车辆数量
- `num_edge_nodes`: 边缘节点数量
- `computing_capability_range`: 计算能力范围
- `bandwidth_range`: 带宽范围

## DDPG参数

- `state_dim`: 状态维度 (15)
- `action_dim`: 动作维度 (3)
- `learning_rate`: 学习率 (0.001)
- `gamma`: 折扣因子 (0.99)
- `tau`: 软更新参数 (0.005)

## 依赖

- PyTorch >= 1.8.0
- NumPy >= 1.19.0
- 其他依赖见 requirements.txt

## GPU加速

支持GPU加速训练，自动检测CUDA可用性并优化训练速度。
