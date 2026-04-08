from pytracking.utils import TrackerParams, FeatureParams
from pytracking.features.extractor import MultiResolutionExtractor
from pytracking.features import deep
import torch


def parameters():
    params = TrackerParams()

    params.debug = 0
    params.visualization = False
    params.use_gpu = True

    shallow_params = TrackerParams()
    deep_params = TrackerParams()

    params.max_image_sample_size = 210 ** 2
    params.min_image_sample_size = 168 ** 2
    params.search_area_scale = 4.2

    params.CG_iter = 4
    params.init_CG_iter = 32
    params.init_GN_iter = 4
    params.post_init_CG_iter = 0
    params.fletcher_reeves = False
    params.standard_alpha = True
    params.CG_forgetting_rate = 75
    params.precond_data_param = 0.3
    params.precond_reg_param = 0.15
    params.precond_proj_param = 35

    shallow_params.learning_rate = 0.03
    deep_params.learning_rate = 0.012
    shallow_params.output_sigma_factor = 1 / 16
    deep_params.output_sigma_factor = 1 / 4

    params.sample_memory_size = 120
    params.train_skipping = 10

    params.scale_factors = 1.02 ** torch.arange(-1, 2).float()
    params.score_upsample_factor = 1
    params.score_fusion_strategy = 'weightedsum'
    shallow_params.translation_weight = 0.45
    deep_params.translation_weight = 0.55

    params.augmentation = {
        'fliplr': True,
        'rotate': [5, -5, 10, -10],
        'blur': [(2, 0.2), (0.2, 2)],
        'shift': [(4, 4), (-4, 4), (4, -4), (-4, -4)],
    }

    deep_params.use_augmentation = True
    shallow_params.use_augmentation = True

    params.update_projection_matrix = False
    params.projection_reg = 5e-8
    shallow_params.compressed_dim = 8
    deep_params.compressed_dim = 24

    params.interpolation_method = 'bicubic'
    params.interpolation_bicubic_a = -0.75
    params.interpolation_centering = True
    params.interpolation_windowing = False

    shallow_params.use_reg_window = True
    shallow_params.reg_window_min = 1e-4
    shallow_params.reg_window_edge = 10e-3
    shallow_params.reg_window_power = 2
    shallow_params.reg_sparsity_threshold = 0.05

    deep_params.use_reg_window = True
    deep_params.reg_window_min = 10e-4
    deep_params.reg_window_edge = 45e-3
    deep_params.reg_window_power = 2
    deep_params.reg_sparsity_threshold = 0.1

    params.preload_features_on_create = True
    params.warmup_on_create = True
    params.warmup_image_sz = 192

    params.adaptive_update = True
    params.adaptive_min_interval = 10
    params.adaptive_max_interval = 20
    params.adaptive_motion_threshold = 0.09
    params.adaptive_scale_threshold = 0.02
    params.adaptive_score_threshold = 0.26
    params.adaptive_cg = True
    params.stable_CG_iter = 2
    params.recovery_CG_iter = 4

    params.scale_refresh_interval = 3
    params.scale_confidence_threshold = 0.22
    params.intermediate_scale_factors = torch.ones(1)
    params.pretrained_projection_path = 'person_ecolite/person_ecolite_person_mix_projection_v1.pth'

    fparams = FeatureParams(feature_params=[shallow_params, deep_params])
    features = deep.PersonEcoLiteFeature(
        output_layers=['stem', 'stage4'],
        net_path='person_ecolite/person_ecolite_person_mix_v1.pth',
        use_gpu=params.use_gpu,
        fparams=fparams,
        pool_stride=[1, 1],
        normalize_power=2,
    )
    features.use_inference_mode = True
    features.use_amp = False
    features.use_channels_last = False

    params.features = MultiResolutionExtractor([features])
    return params
