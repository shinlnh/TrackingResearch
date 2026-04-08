from pytracking.parameter.eco.jetson_fast_trt_rgb import parameters as base_parameters
import torch


def parameters():
    params = base_parameters()
    params.scale_factors = 1.02 ** torch.arange(-1, 2).float()
    params.scale_refresh_interval = 3
    params.scale_confidence_threshold = 0.26
    params.intermediate_scale_factors = torch.ones(1)
    params.search_area_scale = 3.0
    params.features.features[1].engine_path = 'resnet18_vggmconv1/resnet18_vggmconv1_otb_large_b3_fp16.engine'
    return params
