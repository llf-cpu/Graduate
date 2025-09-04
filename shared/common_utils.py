import os
import json
import torch
import numpy as np
from typing import Dict, Any, List

def save_results(results: Dict[str, Any], save_path: str):
    """保存实验结果"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"结果已保存至: {save_path}")

def load_results(load_path: str) -> Dict[str, Any]:
    """加载实验结果"""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"结果文件不存在: {load_path}")
    
    with open(load_path, 'r') as f:
        results = json.load(f)
    
    return results

def calculate_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """计算评估指标"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='weighted')
    recall = recall_score(targets, predictions, average='weighted')
    f1 = f1_score(targets, predictions, average='weighted')
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

def format_time(seconds: float) -> str:
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def print_progress_bar(iteration: int, total: int, prefix: str = '', suffix: str = '', length: int = 50):
    """打印进度条"""
    percent = f"{100 * (iteration / float(total)):.1f}"
    filled_length = int(length * iteration // total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='')
    if iteration == total:
        print()

def check_gpu_availability() -> bool:
    """检查GPU可用性"""
    return torch.cuda.is_available()

def get_device_info() -> Dict[str, Any]:
    """获取设备信息"""
    device_info = {
        'cuda_available': torch.cuda.is_available(),
        'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None
    }
    
    if torch.cuda.is_available():
        device_info['device_name'] = torch.cuda.get_device_name(0)
        device_info['memory_total'] = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
    
    return device_info
