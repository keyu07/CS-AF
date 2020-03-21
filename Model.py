import torchvision.models as models
from torch import nn
import pretrainedmodels as ptm
from efficientnet_pytorch import EfficientNet
import torch

def nets(model, num_class):

    if model == 'inceptionv4':
        model = ptm.inceptionv4(num_classes=1000, pretrained='imagenet')
        model.last_linear = nn.Linear(model.last_linear.in_features, num_class)
        return model

    if model == 'senet154':
        model = ptm.senet154(num_classes=1000, pretrained='imagenet')
        model.last_linear = nn.Linear(model.last_linear.in_features, num_class)
        return model

    if model == 'pnasnet':
        model = ptm.pnasnet5large(num_classes=1000, pretrained='imagenet')
        model.last_linear = nn.Linear(model.last_linear.in_features, num_class)
        return model

    if model == 'xception':
        model = ptm.xception(num_classes=1000, pretrained='imagenet')
        model.last_linear = nn.Linear(model.last_linear.in_features, num_class)
        return model

    if model == 'incepresv2':
        model = ptm.inceptionresnetv2(num_classes=1000, pretrained='imagenet')
        model.last_linear = nn.Linear(model.last_linear.in_features, num_class)
        return model

    if model == 'resnet152':
        model = models.resnet152(pretrained=True)
        model.fc = nn.Linear(2048, num_class)
        return model
        
    if model == 'se_resxt101':
        model = ptm.se_resnext101_32x4d(num_classes=1000, pretrained='imagenet')
        model.last_linear = nn.Linear(model.last_linear.in_features, num_class)
        return model
        
    if model == 'nasnet':
        model = ptm.nasnetalarge(num_classes=1000, pretrained='imagenet')
        model.last_linear = nn.Linear(model.last_linear.in_features, num_class)
        return model
        
    if model == 'dpn': # 224 input size
        model = ptm.dpn107(num_classes=1000, pretrained='imagenet+5k')
        model.last_linear = nn.Conv2d(model.last_linear.in_channels, num_class,
                                      kernel_size=1, bias=True)
        return model
        
    if model == 'resnext101':# 320 input size
        model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x16d_wsl')
        model.fc = nn.Linear(2048, num_class)
        return model  
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
