from pytracking.utils import TrackerParams, FeatureParams
from pytracking.features.extractor import MultiResolutionExtractor
from pytracking.features import deep
import torch


def parameters():
    params = TrackerParams()

    params.debug = 0
    params.visualization = False
    params.use_gpu = torch.cuda.is_available()

    deep_params = TrackerParams()

    # DeepSRDCF is an SRDCF-style tracker using only the first convolutional layer.
    params.max_image_sample_size = 250**2
    params.min_image_sample_size = 200**2
    params.search_area_scale = 4.0

    params.CG_iter = 5
    params.init_CG_iter = 100
    params.init_GN_iter = 10
    params.post_init_CG_iter = 0
    params.fletcher_reeves = False
    params.standard_alpha = True
    params.CG_forgetting_rate = 75
    params.precond_data_param = 0.3
    params.precond_reg_param = 0.15
    params.precond_proj_param = 35

    deep_params.learning_rate = 0.025
    deep_params.output_sigma_factor = 1 / 16

    params.sample_memory_size = 200
    params.train_skipping = 1

    params.scale_factors = 1.01 ** torch.arange(-3, 4).float()
    params.score_upsample_factor = 1
    params.score_fusion_strategy = "sum"
    deep_params.translation_weight = 1.0

    params.augmentation = {}
    deep_params.use_augmentation = False

    # Keep the first-frame PCA basis fixed to stay close to the DeepSRDCF formulation.
    params.update_projection_matrix = False
    params.projection_reg = 5e-8
    deep_params.compressed_dim = 40

    params.interpolation_method = "bicubic"
    params.interpolation_bicubic_a = -0.75
    params.interpolation_centering = True
    params.interpolation_windowing = False

    deep_params.use_reg_window = True
    deep_params.reg_window_min = 1e-4
    deep_params.reg_window_edge = 10e-3
    deep_params.reg_window_power = 2
    deep_params.reg_sparsity_threshold = 0.05

    fparams = FeatureParams(feature_params=[deep_params])
    features = deep.ResNet18m1(
        output_layers=["vggconv1"],
        use_gpu=params.use_gpu,
        fparams=fparams,
        pool_stride=[2],
        normalize_power=2,
    )

    params.features = MultiResolutionExtractor([features])

    return params
