from pytracking.parameter.eco.person_proj_rt_v1 import parameters as base_parameters


def parameters():
    params = base_parameters()
    params.projection_fallback = True
    params.projection_fallback_ratio_threshold = [0.0, 0.54]
    return params
