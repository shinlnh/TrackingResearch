import torch
import torch.nn as nn
from collections import OrderedDict

from .base import Backbone


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, groups=1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=2):
        super().__init__()
        hidden_channels = max(in_channels * expand_ratio, out_channels)
        self.use_residual = stride == 1 and in_channels == out_channels

        self.expand = ConvBNAct(in_channels, hidden_channels, kernel_size=1)
        self.depthwise = ConvBNAct(hidden_channels, hidden_channels, kernel_size=3, stride=stride, groups=hidden_channels)
        self.project = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.expand(x)
        out = self.depthwise(out)
        out = self.project(out)
        if self.use_residual:
            out = out + x
        return self.act(out)


class PersonEcoLite(Backbone):
    def __init__(self, output_layers, num_classes=1000, frozen_layers=()):
        super().__init__(frozen_layers=frozen_layers)
        self.output_layers = output_layers

        self.stem = ConvBNAct(3, 24, kernel_size=3, stride=2)
        self.stage1 = nn.Sequential(
            DepthwiseSeparableBlock(24, 24, stride=1, expand_ratio=2),
            DepthwiseSeparableBlock(24, 32, stride=2, expand_ratio=2),
        )
        self.stage2 = nn.Sequential(
            DepthwiseSeparableBlock(32, 32, stride=1, expand_ratio=2),
            DepthwiseSeparableBlock(32, 48, stride=2, expand_ratio=3),
        )
        self.stage3 = nn.Sequential(
            DepthwiseSeparableBlock(48, 64, stride=1, expand_ratio=3),
            DepthwiseSeparableBlock(64, 96, stride=2, expand_ratio=3),
        )
        self.stage4 = nn.Sequential(
            DepthwiseSeparableBlock(96, 96, stride=1, expand_ratio=2),
            DepthwiseSeparableBlock(96, 96, stride=1, expand_ratio=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(96, num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def _add_output_and_check(self, name, x, outputs, output_layers):
        if name in output_layers:
            outputs[name] = x
        return len(output_layers) == len(outputs)

    def forward(self, x, output_layers=None):
        outputs = OrderedDict()
        output_layers = self.output_layers if output_layers is None else output_layers

        x = self.stem(x)
        if self._add_output_and_check('stem', x, outputs, output_layers):
            return outputs

        x = self.stage1(x)
        if self._add_output_and_check('stage1', x, outputs, output_layers):
            return outputs

        x = self.stage2(x)
        if self._add_output_and_check('stage2', x, outputs, output_layers):
            return outputs

        x = self.stage3(x)
        if self._add_output_and_check('stage3', x, outputs, output_layers):
            return outputs

        x = self.stage4(x)
        if self._add_output_and_check('stage4', x, outputs, output_layers):
            return outputs

        pooled = self.avgpool(x).flatten(1)
        fc = self.fc(pooled)
        if self._add_output_and_check('fc', fc, outputs, output_layers):
            return outputs

        if len(output_layers) == 1 and output_layers[0] == 'default':
            return fc

        raise ValueError('output_layer is wrong.')


def person_ecolite(output_layers=None, path=None, **kwargs):
    valid_layers = {'stem', 'stage1', 'stage2', 'stage3', 'stage4', 'fc'}
    if output_layers is None:
        output_layers = ['default']
    else:
        for layer in output_layers:
            if layer not in valid_layers:
                raise ValueError('Unknown layer: {}'.format(layer))

    model = PersonEcoLite(output_layers, **kwargs)
    if path is not None:
        model.load_state_dict(torch.load(path), strict=False)
    return model
