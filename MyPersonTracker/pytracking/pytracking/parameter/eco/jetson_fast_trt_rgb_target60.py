from pytracking.parameter.eco.jetson_fast_trt_rgb_refine import parameters as base_parameters


def parameters():
    params = base_parameters()

    deep_params = params.features.features[1].fparams.feature_params[0]

    params.search_area_scale = 3.3
    params.init_CG_iter = 10
    deep_params.compressed_dim = 12

    return params
