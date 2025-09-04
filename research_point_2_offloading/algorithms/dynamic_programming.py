import numpy as np
from typing import List, Tuple, Dict

class AdaptiveTaskSplitting:
    """基于动态规划的自适应任务比例分割算法"""
    
    def __init__(self, task_complexity: float, available_nodes: List[Dict]):
        self.task_complexity = task_complexity
        self.available_nodes = available_nodes
        self.split_ratios = []
        
    def calculate_processing_cost(self, task_portion: float, 
                                node_capability: float) -> float:
        """计算处理成本"""
        if node_capability <= 0:
            return float('inf')
        return task_portion * self.task_complexity / node_capability
    
    def calculate_communication_cost(self, task_portion: float, 
                                   bandwidth: float, latency: float) -> float:
        """计算通信成本"""
        data_size = task_portion * self.task_complexity * 0.1  # 假设数据传输比例
        transmission_time = data_size / bandwidth if bandwidth > 0 else float('inf')
        return transmission_time + latency
    
    def dynamic_programming_split(self) -> List[Tuple[int, float]]:
        """使用动态规划进行任务分割"""
        n_nodes = len(self.available_nodes)
        
        # dp[i][j] 表示前i个节点处理任务比例为j/100时的最小总成本
        resolution = 100  # 精度：1%
        dp = np.full((n_nodes + 1, resolution + 1), float('inf'))
        decision = np.full((n_nodes + 1, resolution + 1), -1, dtype=int)
        
        # 初始化：不使用任何节点时，成本为0
        dp[0][0] = 0
        
        for i in range(1, n_nodes + 1):
            node = self.available_nodes[i-1]
            node_capability = node['compute_capability']
            bandwidth = node['bandwidth']
            latency = node['latency']
            
            for j in range(resolution + 1):  # 当前总分配比例
                # 不使用当前节点
                dp[i][j] = dp[i-1][j]
                
                # 尝试使用当前节点处理不同比例的任务
                for k in range(min(j + 1, resolution + 1)):  # 当前节点处理的比例
                    if dp[i-1][j-k] == float('inf'):
                        continue
                        
                    task_portion = k / resolution
                    if task_portion > 0:
                        proc_cost = self.calculate_processing_cost(task_portion, node_capability)
                        comm_cost = self.calculate_communication_cost(task_portion, bandwidth, latency)
                        total_cost = dp[i-1][j-k] + proc_cost + comm_cost
                        
                        if total_cost < dp[i][j]:
                            dp[i][j] = total_cost
                            decision[i][j] = k
        
        # 回溯找到最优分割方案
        optimal_splits = []
        current_ratio = resolution  # 处理100%的任务
        
        for i in range(n_nodes, 0, -1):
            allocated_ratio = decision[i][current_ratio]
            if allocated_ratio > 0:
                optimal_splits.append((i-1, allocated_ratio / resolution))
                current_ratio -= allocated_ratio
        
        return optimal_splits
    
    def get_split_plan(self) -> Dict:
        """获取任务分割计划"""
        splits = self.dynamic_programming_split()
        
        plan = {
            'splits': [],
            'total_cost': 0,
            'local_ratio': 1.0
        }
        
        for node_idx, ratio in splits:
            node_info = self.available_nodes[node_idx].copy()
            node_info['allocated_ratio'] = ratio
            plan['splits'].append(node_info)
            plan['local_ratio'] -= ratio
        
        # 确保本地处理比例不为负数
        plan['local_ratio'] = max(0, plan['local_ratio'])
        
        return plan
