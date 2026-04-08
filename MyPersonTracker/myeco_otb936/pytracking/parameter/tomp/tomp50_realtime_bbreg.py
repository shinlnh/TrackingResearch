from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone


def parameters():
    params = TrackerParams()

    params.debug = 0
    params.visualization = False

    params.use_gpu = True
    params.use_amp = False
    params.allow_tf32 = True
    params.cudnn_benchmark = True
    params.matmul_precision = 'high'
    params.use_torch_compile = False

    # Faster than default, but keep bbox regression for better accuracy.
    params.train_feature_size = 8
    params.feature_stride = 16
    params.image_sample_size = params.train_feature_size * params.feature_stride
    params.search_area_scale = 2.0
    params.border_mode = 'inside_major'
    params.patch_max_scale_change = 1.4

    params.sample_memory_size = 1
    params.learning_rate = 0.01
    params.init_samples_minimum_weight = 0.25
    params.train_skipping = 100

    params.update_classifier = False
    params.fast_cls_only = False
    params.net_opt_iter = 1
    params.net_opt_update_iter = 1
    params.net_opt_hn_iter = 0

    params.window_output = False
    params.use_augmentation = False
    params.augmentation = {}
    params.augmentation_expansion_factor = 1
    params.random_shift_factor = 0

    params.advanced_localization = False
    params.target_not_found_threshold = 0.2
    params.distractor_threshold = 0.8
    params.hard_negative_threshold = 0.5
    params.target_neighborhood_scale = 2.2
    params.dispalcement_scale = 0.8
    params.hard_negative_learning_rate = 0.02
    params.update_scale_when_uncertain = True
    params.conf_ths = 0.9
    params.search_area_rescaling_at_occlusion = False

    params.net = NetWithBackbone(net_path='tomp50.pth.tar', use_gpu=params.use_gpu)

    params.vot_anno_conversion_type = 'preserve_area'
    params.use_gt_box = True
    params.plot_iou = True

    return params
