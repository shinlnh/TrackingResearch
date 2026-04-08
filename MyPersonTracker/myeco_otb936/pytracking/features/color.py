import torch
from pytracking.features.featurebase import FeatureBase


def _maybe_to_feature_device(feat: torch.Tensor, feature: FeatureBase) -> torch.Tensor:
    device = getattr(feature, 'device', None)
    if device is None:
        return feat
    return feat.to(device, non_blocking=True)


class RGB(FeatureBase):
    """RGB feature normalized to [-0.5, 0.5]."""
    def dim(self):
        return 3

    def stride(self):
        return self.pool_stride

    def extract(self, im: torch.Tensor):
        return _maybe_to_feature_device(im/255 - 0.5, self)


class Grayscale(FeatureBase):
    """Grayscale feature normalized to [-0.5, 0.5]."""
    def dim(self):
        return 1

    def stride(self):
        return self.pool_stride

    def extract(self, im: torch.Tensor):
        return _maybe_to_feature_device(torch.mean(im/255 - 0.5, 1, keepdim=True), self)
