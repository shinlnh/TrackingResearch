from pytracking.parameter.eco.embed_v2 import parameters as base_parameters
import torch


def parameters():
    params = base_parameters()

    shallow_params = params.features.features[0].fparams.feature_params[0]
    deep_params = params.features.features[0].fparams.feature_params[1]
    feature = params.features.features[0]

    feature.net_path = 'resnet18_vggmconv1/resnet18_vggmconv1_person_lasot.pth'
    feature.use_inference_mode = True
    feature.use_amp = False

    params.preload_features_on_create = True
    params.warmup_on_create = True
    params.warmup_image_sz = 192

    params.max_image_sample_size = 192 ** 2
    params.min_image_sample_size = 152 ** 2
    params.search_area_scale = 3.9

    params.CG_iter = 3
    params.init_CG_iter = 24
    params.init_GN_iter = 4
    params.post_init_CG_iter = 0

    params.sample_memory_size = 96
    params.train_skipping = 12

    params.scale_factors = 1.02 ** torch.arange(-1, 2).float()
    params.scale_refresh_interval = 3
    params.scale_confidence_threshold = 0.22
    params.intermediate_scale_factors = torch.ones(1)

    params.adaptive_update = True
    params.adaptive_min_interval = 10
    params.adaptive_max_interval = 22
    params.adaptive_motion_threshold = 0.09
    params.adaptive_scale_threshold = 0.02
    params.adaptive_score_threshold = 0.27
    params.adaptive_cg = True
    params.stable_CG_iter = 2
    params.recovery_CG_iter = 4

    shallow_params.learning_rate = 0.022
    deep_params.learning_rate = 0.009
    shallow_params.compressed_dim = 12
    deep_params.compressed_dim = 32

    return params
