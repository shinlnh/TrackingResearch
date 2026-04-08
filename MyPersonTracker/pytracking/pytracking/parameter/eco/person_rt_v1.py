from pytracking.parameter.eco.embed_v2 import parameters as base_parameters


def parameters():
    params = base_parameters()

    feature = params.features.features[0]
    feature.net_path = 'resnet18_vggmconv1/resnet18_vggmconv1.pth'
    feature.use_inference_mode = True
    feature.use_amp = False

    params.preload_features_on_create = True
    params.warmup_on_create = True
    params.warmup_image_sz = 192

    # Reduce first-frame optimization cost. Track-time quality remains close to the
    # verified setup, but short OTB sequences no longer get dominated by init time.
    params.init_CG_iter = 40
    params.init_GN_iter = 5
    params.post_init_CG_iter = 0

    return params
