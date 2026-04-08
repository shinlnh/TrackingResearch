from pytracking.parameter.eco.default import parameters as default_parameters
import torch


def parameters():
    params = default_parameters()

    params.cudnn_benchmark = True
    params.allow_tf32 = True
    params.matmul_precision = 'high'

    params.adaptive_update = True
    params.adaptive_min_interval = 11
    params.adaptive_max_interval = 22
    params.adaptive_motion_threshold = 0.07
    params.adaptive_scale_threshold = 0.012
    params.adaptive_score_threshold = 0.28

    params.adaptive_cg = True
    params.stable_CG_iter = 3
    params.recovery_CG_iter = 4

    params.scale_refresh_interval = 2
    params.scale_confidence_threshold = 0.28
    params.intermediate_scale_factors = 1.02**torch.arange(-1, 2).float()

    feature = params.features.features[0]
    feature.cudnn_benchmark = True
    feature.allow_tf32 = True
    feature.matmul_precision = 'high'
    feature.use_inference_mode = True
    feature.use_channels_last = False
    feature.use_amp = False

    return params
