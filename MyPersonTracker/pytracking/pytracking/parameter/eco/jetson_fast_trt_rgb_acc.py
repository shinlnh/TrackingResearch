from pytracking.parameter.eco.jetson_fast_trt_rgb import parameters as base_parameters


def parameters():
    params = base_parameters()

    rgb_params = params.features.features[0].fparams.feature_params[0]
    deep_params = params.features.features[1].fparams.feature_params[0]
    params.features.features[1].engine_path = 'resnet18_vggmconv1/resnet18_vggmconv1_otb_large_fp16.engine'

    params.max_image_sample_size = 160 ** 2
    params.min_image_sample_size = 144 ** 2
    params.search_area_scale = 3.8
    params.init_CG_iter = 8
    params.sample_memory_size = 32
    params.train_skipping = 20

    rgb_params.translation_weight = 0.30
    deep_params.translation_weight = 0.70
    rgb_params.learning_rate = 0.035
    deep_params.learning_rate = 0.014
    deep_params.compressed_dim = 10

    return params
