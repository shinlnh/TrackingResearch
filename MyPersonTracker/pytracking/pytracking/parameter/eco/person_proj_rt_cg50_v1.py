from pytracking.parameter.eco.person_proj_rt_v1 import parameters as base_parameters


def parameters():
    params = base_parameters()
    params.init_CG_iter = 50
    return params
