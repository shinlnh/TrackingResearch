from pytracking.utils import FeatureParams, TrackerParams
from pytracking.features import color, deep
from pytracking.features.extractor import MultiResolutionExtractor
import torch


def parameters():
    params = TrackerParams()

    params.debug = 0
    params.visualization = False
    params.use_gpu = True

    rgb_params = TrackerParams()
    deep_params = TrackerParams()

    params.max_image_sample_size = 144 ** 2
    params.min_image_sample_size = 128 ** 2
    params.search_area_scale = 3.4

    params.CG_iter = 1
    params.init_CG_iter = 6
    params.init_GN_iter = 1
    params.post_init_CG_iter = 0
    params.fletcher_reeves = False
    params.standard_alpha = True
    params.CG_forgetting_rate = 25
    params.precond_data_param = 0.25
    params.precond_reg_param = 0.10
    params.precond_proj_param = 6

    rgb_params.learning_rate = 0.028
    deep_params.learning_rate = 0.010
    rgb_params.output_sigma_factor = 1 / 18
    deep_params.output_sigma_factor = 1 / 4

    params.sample_memory_size = 32
    params.train_skipping = 18

    params.scale_factors = 1.02 ** torch.arange(-1, 2).float()
    params.score_upsample_factor = 1
    params.score_fusion_strategy = 'weightedsum'
    rgb_params.translation_weight = 0.24
    deep_params.translation_weight = 0.76

    params.augmentation = {}
    rgb_params.use_augmentation = False
    deep_params.use_augmentation = False

    params.update_projection_matrix = False
    params.projection_reg = 5e-8
    rgb_params.compressed_dim = 3
    deep_params.compressed_dim = 10

    params.interpolation_method = 'bicubic'
    params.interpolation_bicubic_a = -0.75
    params.interpolation_centering = True
    params.interpolation_windowing = False

    rgb_params.use_reg_window = True
    rgb_params.reg_window_min = 1e-4
    rgb_params.reg_window_edge = 6e-3
    rgb_params.reg_window_power = 2
    rgb_params.reg_sparsity_threshold = 0.05

    deep_params.use_reg_window = True
    deep_params.reg_window_min = 10e-4
    deep_params.reg_window_edge = 22e-3
    deep_params.reg_window_power = 2
    deep_params.reg_sparsity_threshold = 0.1

    params.cudnn_benchmark = True
    params.allow_tf32 = True
    params.matmul_precision = 'high'

    params.adaptive_update = True
    params.adaptive_min_interval = 16
    params.adaptive_max_interval = 30
    params.adaptive_motion_threshold = 0.10
    params.adaptive_scale_threshold = 0.025
    params.adaptive_score_threshold = 0.24
    params.adaptive_cg = True
    params.stable_CG_iter = 1
    params.recovery_CG_iter = 3

    params.scale_refresh_interval = 4
    params.scale_confidence_threshold = 0.22
    params.intermediate_scale_factors = torch.ones(1)

    params.preload_features_on_create = True
    params.warmup_on_create = True
    params.warmup_image_sz = 128

    rgb_feature = color.RGB(
        fparams=FeatureParams(feature_params=[rgb_params]),
        pool_stride=4,
        normalize_power=2,
    )
    rgb_feature.device = torch.device('cuda')

    deep_feature = deep.ResNet18m1(
        output_layers=['vggconv1'],
        net_path='resnet18_vggmconv1/resnet18_vggmconv1_person_lasot.pth',
        use_gpu=params.use_gpu,
        fparams=FeatureParams(feature_params=[deep_params]),
        pool_stride=[2],
        normalize_power=2,
    )
    deep_feature.use_inference_mode = True
    deep_feature.use_channels_last = False
    deep_feature.use_amp = False

    params.features = MultiResolutionExtractor([rgb_feature, deep_feature])
    return params
