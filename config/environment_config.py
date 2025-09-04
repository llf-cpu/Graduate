# config/environment_config.py

class EnvironmentConfig:
    """环境配置类"""
    
    # 车辆环境配置
    VEHICLE_CONFIG = {
        'num_vehicles': 10,
        'vehicle_speed_range': (20, 80),  # km/h
        'computing_capability_range': (1.0, 3.0),  # GHz
        'battery_capacity': 100,  # Wh
        'communication_range': 200  # meters
    }
    
    # 边缘节点配置
    EDGE_CONFIG = {
        'num_edge_nodes': 5,
        'computing_capability_range': (5.0, 15.0),  # GHz
        'bandwidth_range': (100, 1000),  # Mbps
        'coverage_radius': 500  # meters
    }
    
    # 网络配置
    NETWORK_CONFIG = {
        'latency_range': (10, 100),  # ms
        'packet_loss_rate': 0.01,
        'interference_factor': 0.1
    }
    
    # 任务配置
    TASK_CONFIG = {
        'task_size_range': (1, 50),  # MB
        'task_complexity_range': (1, 10),
        'deadline_range': (100, 1000),  # ms
        'priority_levels': 3
    }
    
    # DDPG配置
    DDPG_CONFIG = {
        'state_dim': 15,
        'action_dim': 3,
        'hidden_dim': 256,
        'learning_rate': 0.001,
        'gamma': 0.99,
        'tau': 0.005,
        'batch_size': 64,
        'memory_size': 10000
    }
