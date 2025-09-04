# config/model_config.py

class ModelConfig:
    """模型配置类"""
    
    # 基础模型配置
    MODEL_TYPES = {
        'yolo_small': {
            'input_size': (416, 416),
            'num_classes': 80,
            'backbone': 'darknet19',
            'anchors': 3
        },
        'yolo_medium': {
            'input_size': (512, 512),
            'num_classes': 80,
            'backbone': 'darknet53',
            'anchors': 3
        },
        'resnet18': {
            'input_size': (224, 224),
            'num_classes': 1000,
            'pretrained': True
        }
    }
    
    # 轻量化配置
    LIGHTWEIGHT_CONFIG = {
        'pruning_ratio': 0.3,
        'quantization_bits': 8,
        'min_compression_ratio': 0.5,
        'max_accuracy_loss': 0.05
    }
    
    # 训练配置
    TRAINING_CONFIG = {
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 100,
        'device': 'auto'  # 'auto', 'cpu', 'cuda'
    }
