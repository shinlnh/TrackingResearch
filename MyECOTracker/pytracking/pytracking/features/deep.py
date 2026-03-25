from pytracking.features.featurebase import FeatureBase, MultiFeatureBase
import torch
import torchvision
from pytracking import TensorList
from pytracking.evaluation.environment import env_settings
import os
from pytracking.utils.loading import load_network
from ltr.models.backbone.resnet18_vggm import resnet18_vggmconv1
from ltr.models.backbone.mobilenetv3 import mobilenet3
from contextlib import nullcontext

normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])


class ResNet18m1(MultiFeatureBase):
    """ResNet18 feature together with the VGG-m conv1 layer.
    args:
        output_layers: List of layers to output.
        net_path: Relative or absolute net path (default should be fine).
        use_gpu: Use GPU or CPU.
    """

    def __init__(self, output_layers, net_path=None, use_gpu=True, *args, **kwargs):
        super(ResNet18m1, self).__init__(*args, **kwargs)

        for l in output_layers:
            if l not in ['vggconv1', 'conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
                raise ValueError('Unknown layer')

        self.output_layers = list(output_layers)
        self.use_gpu = use_gpu
        self.net_path = 'resnet18_vggmconv1/resnet18_vggmconv1.pth' if net_path is None else net_path

    def initialize(self):

        if isinstance(self.pool_stride, int) and self.pool_stride == 1:
            self.pool_stride = [1] * len(self.output_layers)

        self.layer_stride = {'vggconv1': 2, 'conv1': 2, 'layer1': 4, 'layer2': 8, 'layer3': 16, 'layer4': 32,
                             'fc': None}
        self.layer_dim = {'vggconv1': 96, 'conv1': 64, 'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512,
                          'fc': None}

        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)

        if os.path.isabs(self.net_path):
            net_path_full = [self.net_path]
        else:
            root_paths = env_settings().network_path
            if isinstance(root_paths, str):
                root_paths = [root_paths]
            net_path_full = [os.path.join(root, self.net_path) for root in root_paths]

        self.net = None
        for net_path in net_path_full:
            try:
                self.net = resnet18_vggmconv1(self.output_layers, path=net_path)
                break
            except:
                pass
        if self.net is None:
            raise Exception('Did not find network file {}'.format(self.net_path))

        self.device = torch.device('cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu')
        self.use_gpu = self.device.type == 'cuda'

        if self.use_gpu:
            torch.backends.cudnn.benchmark = getattr(self, 'cudnn_benchmark', True)
            allow_tf32 = getattr(self, 'allow_tf32', True)
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32
            if hasattr(self, 'matmul_precision'):
                torch.set_float32_matmul_precision(self.matmul_precision)

        self.net = self.net.to(self.device)
        self.mean = self.mean.to(self.device)
        self.std = self.std.to(self.device)

        if self.use_gpu and getattr(self, 'use_channels_last', False):
            self.net = self.net.to(memory_format=torch.channels_last)
        self.net.eval()

    def dim(self):
        return TensorList([self.layer_dim[l] for l in self.output_layers])

    def stride(self):
        return TensorList([s * self.layer_stride[l] for l, s in zip(self.output_layers, self.pool_stride)])

    def extract(self, im: torch.Tensor):
        im = im.to(self.device, non_blocking=self.use_gpu)
        if self.use_gpu and getattr(self, 'use_channels_last', False):
            im = im.contiguous(memory_format=torch.channels_last)

        im = im / 255
        im -= self.mean
        im /= self.std

        inference_context = torch.inference_mode if getattr(self, 'use_inference_mode', True) else torch.no_grad
        autocast_context = nullcontext
        if self.use_gpu and getattr(self, 'use_amp', False):
            amp_dtype = torch.float16 if getattr(self, 'amp_dtype', 'float16') == 'float16' else torch.bfloat16
            autocast_context = lambda: torch.autocast(device_type='cuda', dtype=amp_dtype)

        with inference_context():
            with autocast_context():
                features = TensorList(self.net(im).values())

        return TensorList([feat.float().contiguous().clone() for feat in features])

class Mobilenet(MultiFeatureBase):
    """ResNet18 feature together with the VGG-m conv1 layer.
    args:
        output_layers: List of layers to output.
        net_path: Relative or absolute net path (default should be fine).
        use_gpu: Use GPU or CPU.
    """

    def __init__(self, output_layers, net_path=None, use_gpu=True, *args, **kwargs):
        super(ResNet18m1, self).__init__(*args, **kwargs)

        for l in output_layers:
            if l not in ['init_conv','layer1', 'layer2', 'layer3', 'layer4', 'layer5','layer6','layer_out']:
                raise ValueError('Unknown layer')

        self.output_layers = list(output_layers)
        self.use_gpu = use_gpu
        self.net_path = 'mobilev3_test.t7' if net_path is None else net_path

    def initialize(self):

        if isinstance(self.pool_stride, int) and self.pool_stride == 1:
            self.pool_stride = [1] * len(self.output_layers)

        self.layer_stride = {'init_conv': 2, 'layer1': 2, 'layer2': 4, 'layer3': 8, 'layer4': 16, 'layer5': 16,'layer6': 32, 'layer_out': 32}
        self.layer_dim = {'init_conv': 16, 'layer1': 16, 'layer2': 24, 'layer3': 40, 'layer4': 80, 'layer5': 112,'layer6': 160, 'layer_out': 960}


        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)

        if os.path.isabs(self.net_path):
            net_path_full = [self.net_path]
        else:
            root_paths = env_settings().network_path
            if isinstance(root_paths, str):
                root_paths = [root_paths]
            net_path_full = [os.path.join(root, self.net_path) for root in root_paths]

        self.net = None
        for net_path in net_path_full:
            try:
                self.net = mobilenet3(self.output_layers, path=net_path_full)
                break
            except:
                pass
        if self.net is None:
            raise Exception('Did not find network file {}'.format(self.net_path))

        self.device = torch.device('cuda' if self.use_gpu and torch.cuda.is_available() else 'cpu')
        self.use_gpu = self.device.type == 'cuda'

        if self.use_gpu:
            torch.backends.cudnn.benchmark = getattr(self, 'cudnn_benchmark', True)
            allow_tf32 = getattr(self, 'allow_tf32', True)
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32
            if hasattr(self, 'matmul_precision'):
                torch.set_float32_matmul_precision(self.matmul_precision)

        self.net = self.net.to(self.device)
        self.mean = self.mean.to(self.device)
        self.std = self.std.to(self.device)

        if self.use_gpu and getattr(self, 'use_channels_last', False):
            self.net = self.net.to(memory_format=torch.channels_last)
        self.net.eval()

    def dim(self):
        return TensorList([self.layer_dim[l] for l in self.output_layers])

    def stride(self):
        return TensorList([s * self.layer_stride[l] for l, s in zip(self.output_layers, self.pool_stride)])

    def extract(self, im: torch.Tensor):
        im = im.to(self.device, non_blocking=self.use_gpu)
        if self.use_gpu and getattr(self, 'use_channels_last', False):
            im = im.contiguous(memory_format=torch.channels_last)

        im = im / 255
        im -= self.mean
        im /= self.std

        inference_context = torch.inference_mode if getattr(self, 'use_inference_mode', True) else torch.no_grad
        autocast_context = nullcontext
        if self.use_gpu and getattr(self, 'use_amp', False):
            amp_dtype = torch.float16 if getattr(self, 'amp_dtype', 'float16') == 'float16' else torch.bfloat16
            autocast_context = lambda: torch.autocast(device_type='cuda', dtype=amp_dtype)

        with inference_context():
            with autocast_context():
                features = TensorList(self.net(im).values())

        return TensorList([feat.float().contiguous().clone() for feat in features])

class ATOMResNet18(MultiFeatureBase):
    """ResNet18 feature with the ATOM IoUNet.
    args:
        output_layers: List of layers to output.
        net_path: Relative or absolute net path (default should be fine).
        use_gpu: Use GPU or CPU.
    """

    def __init__(self, output_layers=('layer3',), net_path='atom_iou', use_gpu=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.output_layers = list(output_layers)
        self.use_gpu = use_gpu
        self.net_path = net_path

    def initialize(self):
        self.net = load_network(self.net_path)

        if self.use_gpu:
            self.net.cuda()
        self.net.eval()

        self.iou_predictor = self.net.bb_regressor

        self.layer_stride = {'conv1': 2, 'layer1': 4, 'layer2': 8, 'layer3': 16, 'layer4': 32, 'classification': 16,
                             'fc': None}
        self.layer_dim = {'conv1': 64, 'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512, 'classification': 256,
                          'fc': None}

        self.iounet_feature_layers = self.net.bb_regressor_layer

        if isinstance(self.pool_stride, int) and self.pool_stride == 1:
            self.pool_stride = [1] * len(self.output_layers)

        self.feature_layers = sorted(list(set(self.output_layers + self.iounet_feature_layers)))

        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)

    def dim(self):
        return TensorList([self.layer_dim[l] for l in self.output_layers])

    def stride(self):
        return TensorList([s * self.layer_stride[l] for l, s in zip(self.output_layers, self.pool_stride)])

    def extract(self, im: torch.Tensor):
        im = im / 255
        im -= self.mean
        im /= self.std

        if self.use_gpu:
            im = im.cuda()

        with torch.no_grad():
            output_features = self.net.extract_features(im, self.feature_layers)

        # Store the raw resnet features which are input to iounet
        self.iounet_backbone_features = TensorList(
            [output_features[layer].clone() for layer in self.iounet_feature_layers])

        # Store the processed features from iounet, just before pooling
        with torch.no_grad():
            self.iounet_features = TensorList(self.iou_predictor.get_iou_feat(self.iounet_backbone_features))

        return TensorList([output_features[layer] for layer in self.output_layers])
