from pytracking.utils import FeatureParams, TrackerParams
from pytracking.features import color, deep_trt
from pytracking.features.extractor import MultiResolutionExtractor
import torch


def parameters():
    params = TrackerParams()

    params.debug = 0
    params.visualization = False
    params.use_gpu = True

    gray_params = TrackerParams()
    deep_params = TrackerParams()

    params.max_image_sample_size = 144 ** 2
    params.min_image_sample_size = 128 ** 2
    params.search_area_scale = 3.2

    params.CG_iter = 1
    params.init_CG_iter = 6
    params.init_GN_iter = 1
    params.post_init_CG_iter = 0
    params.fletcher_reeves = False
    params.standard_alpha = True
    params.CG_forgetting_rate = 20
    params.precond_data_param = 0.25
    params.precond_reg_param = 0.10
    params.precond_proj_param = 6

    gray_params.learning_rate = 0.03
    deep_params.learning_rate = 0.012
    gray_params.output_sigma_factor = 1 / 18
    deep_params.output_sigma_factor = 1 / 4

    params.sample_memory_size = 24
    params.train_skipping = 25

    params.scale_factors = torch.ones(1)
    params.score_upsample_factor = 1
    params.score_fusion_strategy = 'weightedsum'
    gray_params.translation_weight = 0.18
    deep_params.translation_weight = 0.82

    params.augmentation = {}
    gray_params.use_augmentation = False
    deep_params.use_augmentation = False

    params.update_projection_matrix = False
    params.projection_reg = 5e-8
    gray_params.compressed_dim = 1
    deep_params.compressed_dim = 8

    params.interpolation_method = 'bicubic'
    params.interpolation_bicubic_a = -0.75
    params.interpolation_centering = True
    params.interpolation_windowing = False

    gray_params.use_reg_window = True
    gray_params.reg_window_min = 1e-4
    gray_params.reg_window_edge = 6e-3
    gray_params.reg_window_power = 2
    gray_params.reg_sparsity_threshold = 0.05

    deep_params.use_reg_window = True
    deep_params.reg_window_min = 10e-4
    deep_params.reg_window_edge = 25e-3
    deep_params.reg_window_power = 2
    deep_params.reg_sparsity_threshold = 0.1

    params.cudnn_benchmark = True
    params.allow_tf32 = True
    params.matmul_precision = 'high'
    params.adaptive_update = False
    params.preload_features_on_create = True
    params.warmup_on_create = True
    params.warmup_image_sz = 132

    gray_feature = color.Grayscale(
        fparams=FeatureParams(feature_params=[gray_params]),
        pool_stride=4,
        normalize_power=2,
    )
    gray_feature.device = torch.device('cuda')

    deep_feature = deep_trt.ResNet18m1TensorRT(
        output_layers=['vggconv1'],
        engine_path='resnet18_vggmconv1/resnet18_vggmconv1_otb_small_fp16.engine',
        use_gpu=params.use_gpu,
        fparams=FeatureParams(feature_params=[deep_params]),
        pool_stride=[2],
        normalize_power=2,
    )
    deep_feature.use_inference_mode = True
    deep_feature.use_channels_last = False
    deep_feature.use_amp = False

    params.features = MultiResolutionExtractor([gray_feature, deep_feature])
    return params
