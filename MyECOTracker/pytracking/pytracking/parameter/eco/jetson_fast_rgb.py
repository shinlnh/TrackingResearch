from pytracking.utils import FeatureParams, TrackerParams
from pytracking.features import color
from pytracking.features.extractor import MultiResolutionExtractor
import torch


def parameters():
    params = TrackerParams()

    params.debug = 0
    params.visualization = False
    params.use_gpu = False

    rgb_params = TrackerParams()

    # Keep the search grid intentionally small on Nano.
    params.max_image_sample_size = 128 ** 2
    params.min_image_sample_size = 128 ** 2
    params.search_area_scale = 2.8

    # Aggressively shrink online optimization cost.
    params.CG_iter = 1
    params.init_CG_iter = 4
    params.init_GN_iter = 1
    params.post_init_CG_iter = 0
    params.fletcher_reeves = False
    params.standard_alpha = True
    params.CG_forgetting_rate = 20
    params.precond_data_param = 0.25
    params.precond_reg_param = 0.10
    params.precond_proj_param = 1

    rgb_params.learning_rate = 0.04
    rgb_params.output_sigma_factor = 1 / 18

    params.sample_memory_size = 12
    params.train_skipping = 45

    # Skip multi-scale search to keep per-frame cost low.
    params.scale_factors = torch.ones(1)
    params.score_upsample_factor = 1
    params.score_fusion_strategy = 'weightedsum'
    rgb_params.translation_weight = 1.0

    # Remove expensive first-frame augmentation.
    params.augmentation = {}
    rgb_params.use_augmentation = False

    params.update_projection_matrix = False
    params.projection_reg = 1e-7
    rgb_params.compressed_dim = 3

    params.interpolation_method = 'bicubic'
    params.interpolation_bicubic_a = -0.75
    params.interpolation_centering = True
    params.interpolation_windowing = False

    rgb_params.use_reg_window = True
    rgb_params.reg_window_min = 1e-4
    rgb_params.reg_window_edge = 4e-3
    rgb_params.reg_window_power = 2
    rgb_params.reg_sparsity_threshold = 0.05

    params.adaptive_update = False

    fparams = FeatureParams(feature_params=[rgb_params])
    features = color.RGB(fparams=fparams, pool_stride=4, normalize_power=2)
    params.features = MultiResolutionExtractor([features])

    return params
