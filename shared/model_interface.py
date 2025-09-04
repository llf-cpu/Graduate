import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from research_point_1_lightweight.models.vision_model import get_vision_model

class LightweightModelInterface:
    """
    轻量化模型接口，用于研究点二调用研究点一的结果。
    这个接口负责加载、管理和使用由研究点一生成的压缩后模型。
    """
    
    def __init__(self, model_type: str = 'MobileNetV3-Large'):
        self.model = None
        self.model_metrics = None
        self.quantization_info = None
        self.model_type = model_type
        
    def load_model_from_checkpoint(self, checkpoint_path: str):
        """从文件加载轻量化模型检查点"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            print(f"成功从 {checkpoint_path} 加载检查点。")
        except FileNotFoundError:
            print(f"错误: 检查点文件未找到 at {checkpoint_path}")
            raise
        except Exception as e:
            print(f"加载检查点时出错: {e}")
            raise

        # 1. 根据模型类型创建基础模型结构
        # 我们加载一个没有预训练权重的模型，因为我们将加载我们自己的状态字典
        self.model = get_vision_model(pretrained=False)
        
        # 2. 加载模型的状态字典
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print("模型状态字典已成功加载。")
        else:
            raise ValueError("检查点中未找到 'model_state_dict'。")
            
        # 3. 加载相关的元数据
        self.model_metrics = checkpoint.get('final_metrics', {})
        self.quantization_info = checkpoint.get('quantization_results', {})
        
        print("轻量化模型接口初始化完成。")
        
    def get_model_info(self) -> Dict[str, Any]:
        """获取关于已加载模型的详细信息"""
        if self.model is None:
            return {'has_model': False, 'message': '模型未加载。'}
            
        return {
            'has_model': True,
            'model_type': self.model_type,
            'metrics': self.model_metrics,
            'quantization_details': self.quantization_info,
        }
    
    def predict(self, input_data: torch.Tensor) -> torch.Tensor:
        """使用加载的模型进行推理"""
        if self.model is None:
            raise ValueError("模型未加载，无法执行推理。")
            
        self.model.eval()
        with torch.no_grad():
            output = self.model(input_data)
        return output
    
    def get_inference_cost(self) -> Dict[str, float]:
        """
        根据加载模型的度量标准估算推理成本。
        这些成本是理论值，用于任务卸载决策。
        """
        if not self.model_metrics:
            # 如果没有度量信息，返回一个默认的高成本，以避免在决策中被优先选择
            print("警告: 模型度量信息不可用，返回默认的高成本。")
            return {
                'compute_gflops': 10.0,  # GFLOPS
                'memory_mb': 50.0,     # MB
                'estimated_latency_ms': 200.0 # ms
            }
            
        # 从度量标准中提取数据
        flops = self.model_metrics.get('flops', 1e9) # 默认为1 GFLOPs
        size_mb = self.model_metrics.get('size_mb', 20.0) # 默认为20MB
        
        # 简化的成本估算模型
        # 计算成本（GFLOPS）
        compute_gflops = flops / 1e9
        # 内存成本（MB）
        memory_mb = size_mb
        # 估算延迟（ms），假设1 GFLOPs需要10ms
        estimated_latency_ms = compute_gflops * 10 
        
        return {
            'compute_gflops': compute_gflops,
            'memory_mb': memory_mb,
            'estimated_latency_ms': estimated_latency_ms
        }

if __name__ == '__main__':
    # 这是一个示例，展示如何使用这个接口
    
    # 假设研究点一已经运行并生成了模型文件
    # 注意：你需要先运行 research_point_1_lightweight/main.py 来生成这个文件
    checkpoint_file = '../research_point_1_lightweight/outputs/lightweight_model.pth'

    try:
        # 1. 创建接口实例
        model_interface = LightweightModelInterface()
        
        # 2. 加载模型
        model_interface.load_model_from_checkpoint(checkpoint_file)
        
        # 3. 显示模型信息
        model_info = model_interface.get_model_info()
        print("\n--- 模型信息 ---")
        import json
        print(json.dumps(model_info, indent=2))
        
        # 4. 估算推理成本
        inference_cost = model_interface.get_inference_cost()
        print("\n--- 推理成本估算 ---")
        print(json.dumps(inference_cost, indent=2))
        
        # 5. 运行一次推理
        # MobileNetV3-Large需要 (1, 3, 224, 224) 的输入
        dummy_input = torch.randn(1, 3, 224, 224)
        print(f"\n--- 运行推理 ---")
        print(f"输入张量形状: {dummy_input.shape}")
        
        output = model_interface.predict(dummy_input)
        print(f"输出张量形状: {output.shape}")
        print("推理成功。")

    except (FileNotFoundError, ValueError) as e:
        print(f"\n测试失败: {e}")
        print("请确保你已经成功运行了 `research_point_1_lightweight/main.py` 以生成所需的模型文件。")
    except Exception as e:
        print(f"\n发生未知错误: {e}")
