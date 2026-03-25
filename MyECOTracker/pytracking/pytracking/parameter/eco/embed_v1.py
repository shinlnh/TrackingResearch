from pytracking.parameter.eco.default import parameters as default_parameters
import torch


def parameters():
    params = default_parameters()

    params.cudnn_benchmark = True
    params.allow_tf32 = True
    params.matmul_precision = 'high'

    params.adaptive_update = True
    params.adaptive_min_interval = 10
    params.adaptive_max_interval = 18
    params.adaptive_motion_threshold = 0.06
    params.adaptive_scale_threshold = 0.012
    params.adaptive_score_threshold = 0.28

    params.scale_refresh_interval = 2
    params.intermediate_scale_factors = 1.02**torch.arange(-1, 2).float()

    feature = params.features.features[0]
    feature.cudnn_benchmark = True
    feature.allow_tf32 = True
    feature.matmul_precision = 'high'
    feature.use_inference_mode = True
    feature.use_channels_last = True
    feature.use_amp = True
    feature.amp_dtype = 'float16'

    return params
