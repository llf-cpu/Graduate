import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Dict, List, Tuple

class Actor(nn.Module):
    """DDPG Actor网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Actor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 输出范围[-1, 1]
        )
        
    def forward(self, state):
        return self.network(state)

class Critic(nn.Module):
    """DDPG Critic网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.network(x)

class DDPGOffloadingAgent:
    """基于改进DDPG的任务卸载优化算法"""
    
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-3):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 检查GPU可用性
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"DDPG使用设备: {self.device}")
        
        # 网络初始化
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.target_actor = Actor(state_dim, action_dim).to(self.device)
        self.target_critic = Critic(state_dim, action_dim).to(self.device)
        
        # 复制参数到目标网络
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # 经验回放
        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        
        # 超参数
        self.gamma = 0.99  # 折扣因子
        self.tau = 0.005   # 软更新参数
        self.noise_std = 0.1  # 噪声标准差
        
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
            
        if add_noise:
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            action = np.clip(action + noise, -1, 1)
            
        return action
    
    def store_experience(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
    
    def update(self):
        """更新网络参数"""
        if len(self.memory) < self.batch_size:
            return
            
        # 采样batch
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.FloatTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).unsqueeze(1).to(self.device)
        
        # 更新Critic
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, next_actions)
            target_values = rewards + (self.gamma * target_q * ~dones)
        
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_values)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新Actor
        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 软更新目标网络
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)
    
    def soft_update(self, local_model, target_model):
        """软更新目标网络"""
        for target_param, local_param in zip(target_model.parameters(), 
                                           local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + 
                                  (1.0 - self.tau) * target_param.data)
    
    def get_offloading_decision(self, state: Dict) -> Dict:
        """根据状态获取卸载决策"""
        # 将状态字典转换为数组
        state_array = np.array([
            state['network_latency'],
            state['local_cpu_usage'],
            state['battery_level'],
            state['task_priority'],
            state['task_complexity']
        ])
        
        action = self.select_action(state_array, add_noise=False)
        
        # 将动作转换为具体的卸载决策
        decision = {
            'offload_ratio': (action[0] + 1) / 2,  # 转换到[0,1]
            'target_node': int((action[1] + 1) * len(state.get('available_nodes', [1])) / 2),
            'priority_weight': (action[2] + 1) / 2
        }
        
        return decision
