import torch.nn as nn
import torchvision

model = torchvision.models.efficientnet_b4()





for name, param in model.named_parameters():
    print(name, param.requires_grad)
