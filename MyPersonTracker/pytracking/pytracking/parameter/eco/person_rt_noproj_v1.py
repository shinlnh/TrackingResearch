from pytracking.parameter.eco.person_rt_v1 import parameters as base_parameters


def parameters():
    params = base_parameters()
    params.update_projection_matrix = False
    params.init_GN_iter = 1
    return params
