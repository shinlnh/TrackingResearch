from pytracking.parameter.eco.jetson_fast_trt_dual import parameters as base_parameters


def parameters():
    params = base_parameters()
    params.features.features[0].engine_path = 'resnet18_vggmconv1/resnet18_vggmconv1_otb_dual_large_fp16.engine'

    shallow_params = params.features.features[0].fparams.feature_params[0]
    deep_params = params.features.features[0].fparams.feature_params[1]

    params.max_image_sample_size = 160 ** 2
    params.min_image_sample_size = 144 ** 2
    params.search_area_scale = 3.2
    params.init_CG_iter = 8
    params.sample_memory_size = 24
    params.train_skipping = 25

    shallow_params.translation_weight = 0.40
    deep_params.translation_weight = 0.60
    shallow_params.learning_rate = 0.022
    deep_params.learning_rate = 0.011
    shallow_params.compressed_dim = 10
    deep_params.compressed_dim = 14

    return params
