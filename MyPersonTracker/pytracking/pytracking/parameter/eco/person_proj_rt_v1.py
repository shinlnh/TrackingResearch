from pytracking.parameter.eco.person_rt_noproj_v1 import parameters as base_parameters


def parameters():
    params = base_parameters()
    params.pretrained_projection_path = (
        'resnet18_vggmconv1/person_projection_lasot_v1.pth'
    )
    return params
