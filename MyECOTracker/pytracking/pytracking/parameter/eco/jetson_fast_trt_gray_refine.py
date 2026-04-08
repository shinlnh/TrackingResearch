from pytracking.parameter.eco.jetson_fast_trt_gray import parameters as base_parameters


def parameters():
    params = base_parameters()

    gray_params = params.features.features[0].fparams.feature_params[0]
    deep_params = params.features.features[1].fparams.feature_params[0]

    params.CG_iter = 2
    params.init_CG_iter = 8
    params.sample_memory_size = 32
    params.train_skipping = 18

    gray_params.translation_weight = 0.24
    deep_params.translation_weight = 0.76
    gray_params.learning_rate = 0.032
    deep_params.learning_rate = 0.013
    deep_params.compressed_dim = 10

    return params
