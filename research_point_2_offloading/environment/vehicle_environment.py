import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import random

class VehicleEnvironment:
    """车载环境模拟"""
    
    def __init__(self, model_interface=None):
        self.model_interface = model_interface
        self.current_state = {}
        self.step_count = 0
        self.max_steps = 100
        
        # 环境参数
        self.base_latency = 0.05  # 基础延迟50ms
        self.battery_capacity = 100.0
        self.cpu_cores = 4
        
        # 边缘节点配置
        self.edge_nodes = [
            {'id': 0, 'compute_capability': 10.0, 'bandwidth': 100, 'latency': 0.02},
            {'id': 1, 'compute_capability': 15.0, 'bandwidth': 150, 'latency': 0.03},
            {'id': 2, 'compute_capability': 8.0, 'bandwidth': 80, 'latency': 0.015},
        ]
        
    def reset(self) -> Dict[str, Any]:
        """重置环境"""
        self.step_count = 0
        
        self.current_state = {
            'network_latency': random.uniform(0.01, 0.1),
            'local_cpu_usage': random.uniform(0.3, 0.9),
            'battery_level': random.uniform(0.4, 1.0),
            'task_priority': random.uniform(0.1, 1.0),
            'task_complexity': random.uniform(1.0, 10.0),
            'available_nodes': self.edge_nodes.copy(),
            'vehicle_speed': random.uniform(20, 80),  # km/h
            'traffic_density': random.uniform(0.2, 1.0)
        }
        
        return self.current_state
    
    def step(self, action: np.ndarray, split_plan: Optional[Dict] = None) -> Tuple[Dict, float, bool]:
        """执行一步"""
        self.step_count += 1
        
        # 解析动作
        offload_ratio = max(0, min(1, (action[0] + 1) / 2))
        target_node_idx = int(max(0, min(len(self.edge_nodes)-1, 
                                        (action[1] + 1) * len(self.edge_nodes) / 2)))
        priority_weight = max(0, min(1, (action[2] + 1) / 2))
        
        # 计算奖励
        reward = self._calculate_reward(offload_ratio, target_node_idx, 
                                      priority_weight, split_plan)
        
        # 更新状态
        self._update_state(offload_ratio, target_node_idx)
        
        # 检查是否结束
        done = self.step_count >= self.max_steps or self.current_state['battery_level'] <= 0
        
        return self.current_state, reward, done
    
    def _calculate_reward(self, offload_ratio: float, target_node_idx: int, 
                         priority_weight: float, split_plan: Optional[Dict] = None) -> float:
        """计算多目标奖励"""
        
        # 延迟计算
        if offload_ratio > 0:
            target_node = self.edge_nodes[target_node_idx]
            network_latency = self.current_state['network_latency']
            processing_latency = (self.current_state['task_complexity'] * offload_ratio / 
                                 target_node['compute_capability'])
            total_latency = network_latency + processing_latency
        else:
            local_processing = self.current_state['task_complexity'] / (self.cpu_cores * (1 - self.current_state['local_cpu_usage']))
            total_latency = local_processing
            
        # 能耗计算
        local_energy = (1 - offload_ratio) * self.current_state['task_complexity'] * 0.5
        transmission_energy = offload_ratio * 0.2
        total_energy = local_energy + transmission_energy
        
        # 精度计算（轻量化模型的影响）
        base_accuracy = 0.9
        if self.model_interface and self.model_interface.model_metrics:
            compression_ratio = self.model_interface.model_metrics.get('compression_ratio', 1.0)
            accuracy_loss = (1 - compression_ratio) * 0.1
            base_accuracy = max(0.7, base_accuracy - accuracy_loss)
        
        # 分割计划的影响
        if split_plan and split_plan['splits']:
            # 如果使用了分割，精度可能有所提升
            base_accuracy += 0.02
        
        # 多目标奖励函数
        latency_reward = -total_latency * 10  # 延迟惩罚
        energy_reward = -(total_energy / self.current_state['battery_level']) * 5  # 能耗惩罚
        accuracy_reward = base_accuracy * 20  # 精度奖励
        priority_reward = priority_weight * self.current_state['task_priority'] * 5
        
        total_reward = latency_reward + energy_reward + accuracy_reward + priority_reward
        
        return total_reward
    
    def _update_state(self, offload_ratio: float, target_node_idx: int):
        """更新环境状态"""
        
        # 更新电池电量
        energy_consumption = offload_ratio * 0.02 + (1 - offload_ratio) * 0.05
        self.current_state['battery_level'] -= energy_consumption
        self.current_state['battery_level'] = max(0, self.current_state['battery_level'])
        
        # 更新CPU使用率
        cpu_change = random.uniform(-0.1, 0.1)
        self.current_state['local_cpu_usage'] = max(0.2, min(1.0, 
            self.current_state['local_cpu_usage'] + cpu_change))
        
        # 更新网络延迟（模拟移动性）
        latency_change = random.uniform(-0.01, 0.01)
        self.current_state['network_latency'] = max(0.01, min(0.2,
            self.current_state['network_latency'] + latency_change))
        
        # 更新任务复杂度
        self.current_state['task_complexity'] = random.uniform(1.0, 10.0)
        self.current_state['task_priority'] = random.uniform(0.1, 1.0)
    
    def get_state_dimension(self) -> int:
        """获取状态空间维度"""
        return 5  # network_latency, local_cpu_usage, battery_level, task_priority, task_complexity
    
    def get_state_array(self) -> np.ndarray:
        """获取状态数组"""
        return np.array([
            self.current_state['network_latency'],
            self.current_state['local_cpu_usage'],
            self.current_state['battery_level'],
            self.current_state['task_priority'],
            self.current_state['task_complexity']
        ])
    
    def get_current_metrics(self) -> Dict[str, float]:
        """获取当前性能指标"""
        return {
            'latency': self.current_state['network_latency'],
            'energy': 1.0 - self.current_state['battery_level'],
            'accuracy': 0.9,  # 简化的精度估算
            'cpu_usage': self.current_state['local_cpu_usage']
        }