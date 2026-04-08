from pytracking.utils import FeatureParams, TrackerParams
from pytracking.features import deep_trt
from pytracking.features.extractor import MultiResolutionExtractor
import torch


def parameters():
    params = TrackerParams()

    params.debug = 0
    params.visualization = False
    params.use_gpu = True

    shallow_params = TrackerParams()
    deep_params = TrackerParams()

    params.max_image_sample_size = 144 ** 2
    params.min_image_sample_size = 128 ** 2
    params.search_area_scale = 3.0

    params.CG_iter = 1
    params.init_CG_iter = 6
    params.init_GN_iter = 1
    params.post_init_CG_iter = 0
    params.fletcher_reeves = False
    params.standard_alpha = True
    params.CG_forgetting_rate = 20
    params.precond_data_param = 0.25
    params.precond_reg_param = 0.10
    params.precond_proj_param = 8

    shallow_params.learning_rate = 0.02
    deep_params.learning_rate = 0.010
    shallow_params.output_sigma_factor = 1 / 16
    deep_params.output_sigma_factor = 1 / 4

    params.sample_memory_size = 20
    params.train_skipping = 30

    params.scale_factors = torch.ones(1)
    params.score_upsample_factor = 1
    params.score_fusion_strategy = 'weightedsum'
    shallow_params.translation_weight = 0.45
    deep_params.translation_weight = 0.55

    params.augmentation = {}
    shallow_params.use_augmentation = False
    deep_params.use_augmentation = False

    params.update_projection_matrix = False
    params.projection_reg = 5e-8
    shallow_params.compressed_dim = 8
    deep_params.compressed_dim = 12

    params.interpolation_method = 'bicubic'
    params.interpolation_bicubic_a = -0.75
    params.interpolation_centering = True
    params.interpolation_windowing = False

    shallow_params.use_reg_window = True
    shallow_params.reg_window_min = 1e-4
    shallow_params.reg_window_edge = 8e-3
    shallow_params.reg_window_power = 2
    shallow_params.reg_sparsity_threshold = 0.05

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

    fparams = FeatureParams(feature_params=[shallow_params, deep_params])
    features = deep_trt.ResNet18m1TensorRT(
        output_layers=['vggconv1', 'layer3'],
        engine_path='resnet18_vggmconv1/resnet18_vggmconv1_otb_dual_small_fp16.engine',
        use_gpu=params.use_gpu,
        fparams=fparams,
        pool_stride=[2, 1],
        normalize_power=2,
    )
    features.use_inference_mode = True
    features.use_channels_last = False
    features.use_amp = False

    params.features = MultiResolutionExtractor([features])
    return params
