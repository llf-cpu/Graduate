import torch
import torch.nn as nn
from typing import Dict, Any

class YOLOSmall(nn.Module):
    """简化的YOLO检测模型"""
    
    def __init__(self, num_classes: int = 10):
        super(YOLOSmall, self).__init__()
        
        self.backbone = nn.Sequential(
            # 第一层
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # 第二层
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # 第三层
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # 第四层
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

def create_detection_model(model_type: str = 'yolo_small', **kwargs) -> nn.Module:
    """创建检测模型"""
    if model_type == 'yolo_small':
        return YOLOSmall(**kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")