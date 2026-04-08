from pytracking.parameter.eco.embed_v2 import parameters as base_parameters
import torch


def parameters():
    params = base_parameters()

    feature = params.features.features[0]
    feature.net_path = 'resnet18_vggmconv1/resnet18_vggmconv1_person_lasot.pth'
    feature.use_inference_mode = True
    feature.use_amp = False

    params.preload_features_on_create = True
    params.warmup_on_create = True
    params.warmup_image_sz = 224

    params.search_area_scale = 4.6
    params.sample_memory_size = 180
    params.train_skipping = 8

    params.adaptive_update = True
    params.adaptive_min_interval = 8
    params.adaptive_max_interval = 18
    params.adaptive_motion_threshold = 0.10
    params.adaptive_scale_threshold = 0.025
    params.adaptive_score_threshold = 0.28

    params.adaptive_cg = True
    params.stable_CG_iter = 2
    params.recovery_CG_iter = 5

    params.scale_refresh_interval = 2
    params.scale_confidence_threshold = 0.25
    params.intermediate_scale_factors = torch.ones(1)

    return params
