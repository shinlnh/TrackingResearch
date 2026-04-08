from pytracking.parameter.eco.person_proj_rt_v1 import parameters as base_parameters


def parameters():
    params = base_parameters()
    params.scale_refresh_interval = 2
    params.scale_confidence_threshold = 0.24
    return params
