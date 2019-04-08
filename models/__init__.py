from .AlexNet import AlexNet
from .ResNet34 import ResNet34
# 添加以上引用 这样在主函数中就可以写成
# from models import AlexNet
# 或
# import models
# model = models.AlexNet()
# 或
# import models
# model = getattr(models,'AlexNet')()
# 其中最后一种写法最为关键，这意味着我们可以通过字符串直接指定使用的模型，而不必使用判断语句，也不必在每次新增加模型后都修改代码。
# 新增模型后只需要在models/__init__.py中加上
# from .new_module import NewModule


# 其他模型定义的注意事项：
# 尽量使用nn.Sequential (如Alexnet)
# 将经常使用的结构封装成子Module（比如GoogLeNet的Inception结构，ResNet的Residual Block结构）
# 将重复且有规律性的结构，用函数生成（比如VGG的多种变体，ResNet多种变体都是由多个重复卷积层组成）
# 读者可以看看在`models/resnet34.py`如何用不到80行的代码(包括空行和注释)实现resnet34。当然这些模型在torchvision中有实现，而且还提供了预训练的权重，读者可以很方便的使用：

#     import torchvision as tv
#     resnet34 = tv.models.resnet34(pretrained=True)