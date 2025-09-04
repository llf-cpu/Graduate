import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

def get_vision_model(pretrained=True):
    """
    获取一个预训练的MobileNetV3-Large模型。

    Args:
        pretrained (bool): 如果为True，返回在ImageNet上预训练的模型。

    Returns:
        torch.nn.Module: MobileNetV3-Large模型。
    """
    if pretrained:
        weights = MobileNet_V3_Large_Weights.DEFAULT
        model = mobilenet_v3_large(weights=weights)
    else:
        model = mobilenet_v3_large(weights=None)
    
    # 我们可以根据需要修改模型的最后几层，以适应特定的任务
    # 例如，如果我们的任务不是1000类的ImageNet分类
    # num_ftrs = model.classifier[3].in_features
    # model.classifier[3] = nn.Linear(num_ftrs, num_classes) # num_classes是你的任务类别数

    return model

if __name__ == '__main__':
    # 创建模型实例
    model = get_vision_model(pretrained=True)
    model.eval()
    
    # 打印模型结构
    print(model)

    # 创建一个虚拟输入张量
    # MobileNetV3-Large的默认输入尺寸是 (3, 224, 224)
    dummy_input = torch.randn(1, 3, 224, 224)

    # 测试模型前向传播
    with torch.no_grad():
        output = model(dummy_input)

    print(f"\n模型加载成功。")
    print(f"输入张量形状: {dummy_input.shape}")
    print(f"输出张量形状: {output.shape}")

    # 计算模型参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params / 1e6:.2f} M")
    print(f"可训练参数量: {trainable_params / 1e6:.2f} M")
