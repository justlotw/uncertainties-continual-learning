import torch.nn as nn

from torchvision.models import (
    resnet18, resnet34, resnet50, 
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights,
    vgg11, vgg13, vgg16,
    VGG11_Weights, VGG13_Weights, VGG16_Weights,
    efficientnet_b1,
)

from .effnet import EfficientNet

from .utils import weights_init

VALID_MODEL_NAMES = {
    'resnet18': 'resnet',
    'resnet34': 'resnet',
    'resnet50': 'resnet',
    'vgg11': 'vgg',
    'vgg13': 'vgg',
    'vgg16': 'vgg',
    'efficientnet-b0': 'efficientnet',
    'efficientnet-b1': 'efficientnet',
    'custom_cnn': 'custom_cnn',
    'small_custom_cnn': 'custom_cnn',
}

EFFICIENTNET_MODELS = {
    'efficientnet-b0': EfficientNet.from_name('efficientnet-b0'),
    'efficientnet-b1': EfficientNet.from_name('efficientnet-b1'),
}

RESNET_MODELS = {
    'resnet18': (resnet18, ResNet18_Weights.DEFAULT),
    'resnet34': (resnet34, ResNet34_Weights.DEFAULT),
    'resnet50': (resnet50, ResNet50_Weights.DEFAULT),
}

RESNET_FREEZE_LAYERS = {
    0: 'all',
    1: 'exclude_last',
    2: 'exclude_last_2',
}

VGG_MODELS = {
    'vgg11': (vgg11, VGG11_Weights.DEFAULT),
    'vgg13': (vgg13, VGG13_Weights.DEFAULT),
    'vgg16': (vgg16, VGG16_Weights.DEFAULT),
}

VGG_FREEZE_LAYERS = {
    0: 'all',
    1: 'cnn',
    2: 'cnn_exclude_last_2',
}

## MODELS ##
def get_conv_model(model_name, num_classes=10, pretrained=False, **kwargs):
    model_name = model_name.lower()
    if model_name not in VALID_MODEL_NAMES:
        raise ValueError('model_name must be one of {}'.format(VALID_MODEL_NAMES.keys()))
    model_type = VALID_MODEL_NAMES[model_name]
    if model_type == 'resnet':
        return _get_resnet_model(model_name, num_classes, pretrained, **kwargs)
    elif model_type == 'vgg':
        return _get_vgg_model(model_name, num_classes, pretrained, **kwargs)
    elif model_type == 'custom_cnn':
        if model_name == 'small_custom_cnn':
            return CustomCNN(num_classes, input_shape=56, cnn_channels=[16, 32], fc_channels=[512, 128, 64])
        cnn_channels = kwargs.get('cnn_channels', None)
        fc_channels = kwargs.get('fc_channels', None)
        if cnn_channels is None:
            if fc_channels is None:
                return CustomCNN(num_classes, input_shape=224)
            else:
                return CustomCNN(num_classes, input_shape=224, fc_channels=fc_channels)
        else:
            if fc_channels is None:
                return CustomCNN(num_classes, input_shape=224, cnn_channels=cnn_channels)
        return CustomCNN(num_classes, input_shape=224, cnn_channels=cnn_channels, fc_channels=fc_channels)
    elif model_type == 'efficientnet':
        return _get_efficientnet_model(model_name, num_classes=num_classes)
    else:
        raise ValueError('model_type must be one of {}'.format(VALID_MODEL_NAMES.values()))

def _get_efficientnet_model(model_name, num_classes):
    if model_name not in EFFICIENTNET_MODELS:
        raise ValueError('model_name must be one of {}'.format(EFFICIENTNET_MODELS))
    effnet = EFFICIENTNET_MODELS[model_name]
    in_features = effnet._fc.out_features
    model = nn.Sequential(
        effnet,
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes)
    )
    return model

def _get_resnet_model(model_name, num_classes, pretrained, **kwargs):
    if model_name not in RESNET_MODELS:
        raise ValueError('model_name must be one of {}'.format(RESNET_MODELS.keys()))
    model, weights = RESNET_MODELS[model_name]
    if pretrained:
        model = model(weights=weights)
        layers_to_freeze = kwargs.get('layers_to_freeze', 'all')
        layers_to_freeze = RESNET_FREEZE_LAYERS.get(layers_to_freeze, layers_to_freeze)
        reset_weight = kwargs.get('reset_weights', False)
        if layers_to_freeze == 'all':
            for param in model.parameters():
                param.requires_grad = False
        elif layers_to_freeze == 'exclude_last':
            for param in model.parameters():
                param.requires_grad = False
            for param in model.layer4.parameters():
                param.requires_grad = True
            if reset_weight:
                model.layer4.apply(weights_init)
        elif layers_to_freeze == 'exclude_last_2':
            for param in model.parameters():
                param.requires_grad = False
            for param in model.layer3.parameters():
                param.requires_grad = True
            for param in model.layer4.parameters():
                param.requires_grad = True
            if reset_weight:
                model.layer3.apply(weights_init)
                model.layer4.apply(weights_init)
        else:
            raise ValueError(f"layers_to_freeze must be one of {['all', 'exclude_last', 'exclude_last_2']}")
    else:
        model = model()

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model

def _get_vgg_model(model_name, num_classes, pretrained, **kwargs):
    if model_name not in VGG_MODELS:
        raise ValueError('model_name must be one of {}'.format(VGG_MODELS.keys()))
    model, weights = VGG_MODELS[model_name]
    if pretrained:
        layers_to_freeze = kwargs.get('layers_to_freeze', 'all')
        layers_to_freeze = VGG_FREEZE_LAYERS.get(layers_to_freeze, layers_to_freeze)
        reset_weight = kwargs.get('reset_weights', False)
        model = model(weights=weights)
        if layers_to_freeze == 'all':
            for param in model.parameters():
                param.requires_grad = False
        elif layers_to_freeze == 'cnn':
            for param in model.features.parameters():
                param.requires_grad = False
            if reset_weight:
                model.classifier.apply(weights_init)
        elif layers_to_freeze == 'cnn_exclude_last_2':
            for param in model.features[:-5].parameters():
                param.requires_grad = False
            if reset_weight:
                model.features[-5:].apply(weights_init)
        else:
            raise ValueError(f"layers_to_freeze must be one of {['all', 'cnn', 'cnn_exclude_last_2']}")
    else:
        model = model()

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


### CUSTOM MODELS ###
class CustomCNN(nn.Module):
    def __init__(self, num_classes, input_shape, cnn_channels=[32, 64, 128, 128, 256], fc_channels=[4096, 1024]):
        super(CustomCNN, self).__init__()
        cnn_channels = [3] + cnn_channels
        self.num_cnn_layers = len(cnn_channels)
        self.num_fc_layers = len(fc_channels)
        convolution_layers = []
        for (c_in, c_out) in zip(cnn_channels[:-1], cnn_channels[1:]):
            convolution_layers.append(nn.Conv2d(c_in, c_out, 3, padding='same'))
        self.convolution_layers = nn.ModuleList(convolution_layers)
        fc_layers = []
        fc_in = cnn_channels[-1] * (input_shape // (2 ** (self.num_cnn_layers - 1))) ** 2
        for fc_out in fc_channels:
            fc_layers.append(nn.Linear(fc_in, fc_out))
            fc_in = fc_out
        self.fc_layers = nn.ModuleList(fc_layers)
        self.fc_out = nn.Linear(fc_channels[-1], num_classes)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        for layer in self.convolution_layers:
            x = self.relu(layer(x))
            x = self.pool(x)
        x = x.view(x.size(0), -1)
        for layer in self.fc_layers:
            x = self.relu(layer(x))
            x = self.dropout(x)
        x = self.fc_out(x)
        return x