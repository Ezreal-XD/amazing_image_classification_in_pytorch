import torch
import torch.nn as nn

__all__ = [
    'ShapeNet', 'shapenet9', 'shapenet9_bn', 'shapenet11', 'shapenet11_bn', 'shapenet13', 'shapenet13_bn',
    'shapenet15_bn', 'shapenet15',
]


class ShapeNet(nn.Module):

    def __init__(self, features, num_classes=10, init_weights=True):
        super(ShapeNet, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 2 * 2, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class ShapeNet_gap(nn.Module):

    def __init__(self, features, classes=10, init_weights=True):
        super(ShapeNet_gap, self).__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        self.gap = nn.AvgPool2d(2)
        self.classifier = nn.Sequential(
            nn.Linear(150, classes, bias=False),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                # nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M'],
    'GAP300': [64, 'M', 128, 'M', 256, 'M', 512, 300, 'M'],
    'GAP': [64, 'M', 128, 'M', 256, 'M', 512, 150, 'M'],

}


def _shapenet(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = ShapeNet(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = ""
        model.load_state_dict(state_dict)
    return model


def _shapenet_gap(arch, cfg, batch_norm, pretrained, progress, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = ShapeNet_gap(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    if pretrained:
        state_dict = ""
        model.load_state_dict(state_dict)
    return model


def shapenet9(pretrained=False, progress=True, **kwargs):
    r"""ShapeNet 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shapenet('shapenet11', 'A', False, pretrained, progress, **kwargs)


def shapenet9_bn(pretrained=False, progress=True, **kwargs):
    r"""ShapeNet 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shapenet('shapenet11_bn', 'A', True, pretrained, progress, **kwargs)


def shapenet11(pretrained=False, progress=True, **kwargs):
    r"""ShapeNet 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shapenet('shapenet13', 'B', False, pretrained, progress, **kwargs)


def shapenet11_bn(pretrained=False, progress=True, **kwargs):
    r"""ShapeNet 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shapenet('shapenet13_bn', 'B', True, pretrained, progress, **kwargs)


def shapenet13(pretrained=False, progress=True, **kwargs):
    r"""ShapeNet 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shapenet('shapenet16', 'D', False, pretrained, progress, **kwargs)


def shapenet13_bn(pretrained=False, progress=True, **kwargs):
    r"""ShapeNet 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shapenet('shapenet16_bn', 'D', True, pretrained, progress, **kwargs)


def shapenet15(pretrained=False, progress=True, **kwargs):
    r"""ShapeNet 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shapenet('shapenet19', 'E', False, pretrained, progress, **kwargs)


def shapenet15_bn(pretrained=False, progress=True, **kwargs):
    r"""ShapeNet 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shapenet('shapenet19_bn', 'E', True, pretrained, progress, **kwargs)


def shapenet15_bn_gap(pretrained=False, progress=True, **kwargs):
    r"""ShapeNet 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shapenet_gap('shapenet19_bn', 'GAP', True, pretrained, progress, **kwargs)


def shapenet15_bn_gap2(pretrained=False, progress=True, **kwargs):
    r"""ShapeNet 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _shapenet_gap('shapenet19_bn', 'GAP2', True, pretrained, progress, **kwargs)

