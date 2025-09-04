import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import random
from shared.model_interface import LightweightModelInterface

class VehicleEnvironment:
    """
    车载环境模拟器。
    这个类模拟了车辆的动态状态、网络条件以及与边缘节点的交互。
    它使用来自 `LightweightModelInterface` 的模型信息来计算更真实的性能指标。
    """
    
    def __init__(self, model_interface: Optional[LightweightModelInterface] = None):
        self.model_interface = model_interface
        self.current_state = {}
        self.step_count = 0
        self.max_steps = 100
        
        # 环境参数
        self.base_latency = 0.05  # 基础延迟50ms
        self.battery_capacity = 100.0
        self.cpu_cores = 4
        
        # 性能指标记录
        self.current_latency = 0.0
        self.current_energy = 0.0
        self.current_accuracy = 0.95 # 默认高精度

        # 边缘节点配置
        self.edge_nodes = [
            {'id': 0, 'compute_capability': 20.0, 'bandwidth': 100, 'latency': 0.02}, # GFLOPS
            {'id': 1, 'compute_capability': 30.0, 'bandwidth': 150, 'latency': 0.03},
            {'id': 2, 'compute_capability': 15.0, 'bandwidth': 80, 'latency': 0.015},
        ]
        
    def reset(self) -> Dict[str, Any]:
        """重置环境到一个新的随机状态"""
        self.step_count = 0
        
        self.current_state = {
            'network_latency': random.uniform(0.01, 0.1), # s
            'local_cpu_usage': random.uniform(0.3, 0.9),
            'battery_level': random.uniform(0.6, 1.0),
            'task_priority': random.uniform(0.1, 1.0),
            'task_data_size': random.uniform(1.0, 5.0), # MB
            'available_nodes': self.edge_nodes.copy(),
            'vehicle_speed': random.uniform(20, 80),  # km/h
            'traffic_density': random.uniform(0.2, 1.0)
        }
        
        # 使用模型接口获取计算复杂度
        if self.model_interface and self.model_interface.get_model_info()['has_model']:
            model_cost = self.model_interface.get_inference_cost()
            self.current_state['task_complexity'] = model_cost['compute_gflops']
            # 假设精度与模型有关，但这里简化
            self.current_accuracy = 0.95 - (1.0 - model_cost.get('model_quality', 1.0)) * 0.1
        else:
            self.current_state['task_complexity'] = random.uniform(1.0, 5.0) # GFLOPS
            self.current_accuracy = 0.9

        return self.current_state
    
    def step(self, action: np.ndarray, split_plan: Optional[Dict] = None) -> Tuple[Dict, float, bool]:
        """执行一个动作并返回结果"""
        self.step_count += 1
        
        # 1. 解析动作
        offload_ratio = np.clip((action[0] + 1) / 2, 0, 1)
        target_node_idx = int(np.clip((action[1] + 1) / 2 * len(self.edge_nodes), 0, len(self.edge_nodes)-1))
        priority_weight = np.clip((action[2] + 1) / 2, 0, 1)
        
        # 2. 计算奖励和性能指标
        reward = self._calculate_reward_and_metrics(offload_ratio, target_node_idx, priority_weight)
        
        # 3. 更新环境状态
        self._update_state()
        
        # 4. 检查是否结束
        done = self.step_count >= self.max_steps or self.current_state['battery_level'] <= 0.1
        
        return self.current_state, reward, done
    
    def _calculate_reward_and_metrics(self, offload_ratio: float, target_node_idx: int, priority_weight: float) -> float:
        """计算多目标奖励函数，并更新性能指标"""
        
        task_complexity = self.current_state['task_complexity']
        task_data_size = self.current_state['task_data_size']
        
        # --- 计算延迟 ---
        # 本地计算延迟
        local_compute_latency = (task_complexity * (1 - offload_ratio)) / (self.cpu_cores * (1 - self.current_state['local_cpu_usage']) + 1e-5)
        
        # 卸载延迟
        offload_latency = 0
        if offload_ratio > 0:
            target_node = self.edge_nodes[target_node_idx]
            # 传输延迟
            transmission_latency = task_data_size / target_node['bandwidth'] + self.current_state['network_latency']
            # 边缘计算延迟
            edge_compute_latency = (task_complexity * offload_ratio) / target_node['compute_capability']
            offload_latency = transmission_latency + edge_compute_latency

        total_latency = local_compute_latency + offload_latency
        self.current_latency = total_latency * 1000 # ms

        # --- 计算能耗 ---
        # 本地计算能耗 (假设单位)
        local_energy = (1 - offload_ratio) * task_complexity * 0.6 
        # 传输能耗
        transmission_energy = offload_ratio * task_data_size * 0.3
        total_energy = local_energy + transmission_energy
        self.current_energy = total_energy

        # --- 多目标奖励函数 ---
        # 目标: 最小化延迟和能耗，同时考虑任务优先级
        latency_penalty = -total_latency * (1 + self.current_state['task_priority'] * priority_weight)
        energy_penalty = -total_energy * 0.5
        
        # 简单的奖励，鼓励完成任务
        completion_reward = 10.0
        
        total_reward = completion_reward + latency_penalty + energy_penalty
        
        return total_reward
    
    def _update_state(self):
        """在每个步骤后更新环境的动态状态"""
        
        # 更新电池电量
        self.current_state['battery_level'] -= self.current_energy * 0.01 # 缩放因子
        self.current_state['battery_level'] = max(0, self.current_state['battery_level'])
        
        # 模拟状态的随机变化
        self.current_state['local_cpu_usage'] = np.clip(self.current_state['local_cpu_usage'] + random.uniform(-0.1, 0.1), 0.2, 1.0)
        self.current_state['network_latency'] = np.clip(self.current_state['network_latency'] + random.uniform(-0.01, 0.01), 0.01, 0.2)
        
        # 生成新任务
        self.current_state['task_data_size'] = random.uniform(1.0, 5.0)
        self.current_state['task_priority'] = random.uniform(0.1, 1.0)
        if not (self.model_interface and self.model_interface.get_model_info()['has_model']):
             self.current_state['task_complexity'] = random.uniform(1.0, 5.0)

    def get_state_dimension(self) -> int:
        """返回状态向量的维度"""
        return len(self.get_state_array())
    
    def get_state_array(self) -> np.ndarray:
        """将当前状态字典转换为一个numpy数组"""
        state = self.current_state
        return np.array([
            state['network_latency'],
            state['local_cpu_usage'],
            state['battery_level'],
            state['task_priority'],
            state['task_data_size'],
            state['task_complexity']
        ])
    
    # 用于外部查询的接口
    def get_current_latency(self) -> float:
        return self.current_latency

    def get_current_energy(self) -> float:
        return self.current_energy

    def get_current_accuracy(self) -> float:
        return self.current_accuracy