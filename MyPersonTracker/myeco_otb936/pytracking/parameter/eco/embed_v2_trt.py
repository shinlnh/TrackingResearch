from pytracking.features.extractor import MultiResolutionExtractor
from pytracking.features import deep_trt
from pytracking.parameter.eco.default import parameters as default_parameters
import torch


def parameters():
    params = default_parameters()

    params.cudnn_benchmark = True
    params.allow_tf32 = True
    params.matmul_precision = 'high'

    params.adaptive_update = True
    params.adaptive_min_interval = 12
    params.adaptive_max_interval = 24
    params.adaptive_motion_threshold = 0.08
    params.adaptive_scale_threshold = 0.015
    params.adaptive_score_threshold = 0.30

    params.adaptive_cg = True
    params.stable_CG_iter = 2
    params.recovery_CG_iter = 5

    params.scale_refresh_interval = 3
    params.scale_confidence_threshold = 0.26
    params.intermediate_scale_factors = torch.ones(1)

    feature_params = params.features.features[0].fparams
    features = deep_trt.ResNet18m1TensorRT(
        output_layers=['vggconv1', 'layer3'],
        use_gpu=params.use_gpu,
        fparams=feature_params,
        pool_stride=[2, 1],
        normalize_power=2,
    )
    features.cudnn_benchmark = True
    features.allow_tf32 = True
    features.matmul_precision = 'high'
    features.use_inference_mode = True
    features.use_channels_last = False
    features.use_amp = False

    params.features = MultiResolutionExtractor([features])
    return params
