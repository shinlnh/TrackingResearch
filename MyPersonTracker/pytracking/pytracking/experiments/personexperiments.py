from pytracking.evaluation import Tracker, get_dataset

PERSON_ECO_PARAMETER = 'verified_person_v1'
PERSON_ECO_RUN_ID = 1101
PERSON_ECO_DISPLAY_NAME = 'MyPersonTracker-ECO'
PERSON_FAST_ECO_PARAMETER = 'verified_person_fast_v1'
PERSON_FAST_ECO_RUN_ID = 1106
PERSON_FAST_ECO_DISPLAY_NAME = 'MyPersonTracker-ECO-Fast'
PERSON_BALANCE_ECO_PARAMETER = 'verified_person_balance_v1'
PERSON_BALANCE_ECO_RUN_ID = 1107
PERSON_BALANCE_ECO_DISPLAY_NAME = 'MyPersonTracker-ECO-Balance'
PERSON_BALANCE_BASE_ECO_PARAMETER = 'verified_person_balance_base_v1'
PERSON_BALANCE_BASE_ECO_RUN_ID = 1108
PERSON_BALANCE_BASE_ECO_DISPLAY_NAME = 'MyPersonTracker-ECO-Balance-Base'
PERSON_RT_ECO_PARAMETER = 'verified_person_rt_v1'
PERSON_RT_ECO_RUN_ID = 1111
PERSON_RT_ECO_DISPLAY_NAME = 'MyPersonTracker-ECO-RT'
PERSON_RT_NOPROJ_ECO_PARAMETER = 'verified_person_rt_noproj_v1'
PERSON_RT_NOPROJ_ECO_RUN_ID = 1112
PERSON_RT_NOPROJ_ECO_DISPLAY_NAME = 'MyPersonTracker-ECO-RT-NoProj'
PERSON_PROJ_RT_ECO_PARAMETER = 'verified_person_proj_rt_v1'
PERSON_PROJ_RT_ECO_RUN_ID = 1113
PERSON_PROJ_RT_ECO_DISPLAY_NAME = 'MyPersonTracker-ECO-ProjRT'
PERSON_PROJ_RT_PLUS_ECO_PARAMETER = 'verified_person_proj_rt_plus_v1'
PERSON_PROJ_RT_PLUS_ECO_RUN_ID = 1114
PERSON_PROJ_RT_PLUS_ECO_DISPLAY_NAME = 'MyPersonTracker-ECO-ProjRT-Plus'
PERSON_PROJ_RT_CG50_ECO_PARAMETER = 'verified_person_proj_rt_cg50_v1'
PERSON_PROJ_RT_CG50_ECO_RUN_ID = 1115
PERSON_PROJ_RT_CG50_ECO_DISPLAY_NAME = 'MyPersonTracker-ECO-ProjRT-CG50'
PERSON_PROJ_RT_FB54_ECO_PARAMETER = 'verified_person_proj_rt_fb54_v1'
PERSON_PROJ_RT_FB54_ECO_RUN_ID = 1116
PERSON_PROJ_RT_FB54_ECO_DISPLAY_NAME = 'MyPersonTracker-ECO-ProjRT-FB54'
PERSON_LITE_PROJ_RT_ECO_PARAMETER = 'verified_person_lite_proj_rt_v1'
PERSON_LITE_PROJ_RT_ECO_RUN_ID = 1120
PERSON_LITE_PROJ_RT_ECO_DISPLAY_NAME = 'MyPersonTracker-ECO-LiteProjRT'

OTB_PERSON_SEQUENCES = {
    'Basketball', 'BlurBody', 'BlurFace', 'Bolt', 'Bolt2', 'Boy', 'ClifBar', 'Couple', 'Crossing',
    'Crowds', 'Dancer', 'Dancer2', 'David', 'David2', 'David3', 'DragonBaby', 'Dudek', 'FaceOcc1',
    'FaceOcc2', 'FleetFace', 'Freeman1', 'Freeman3', 'Freeman4', 'Girl', 'Girl2', 'Gym', 'Human2',
    'Human3', 'Human4_2', 'Human5', 'Human6', 'Human7', 'Human8', 'Human9', 'Ironman', 'Jogging_1',
    'Jogging_2', 'Jump', 'Jumping', 'Man', 'Mhyang', 'Singer1', 'Singer2', 'Skater', 'Skater2',
    'Skating1', 'Skating2_1', 'Skating2_2', 'Skiing', 'Soccer', 'Subway', 'Surfer', 'Walking',
    'Walking2', 'Woman'
}


def _tracker():
    return [Tracker('eco', PERSON_ECO_PARAMETER, PERSON_ECO_RUN_ID, PERSON_ECO_DISPLAY_NAME)]


def _fast_tracker():
    return [Tracker('eco', PERSON_FAST_ECO_PARAMETER, PERSON_FAST_ECO_RUN_ID, PERSON_FAST_ECO_DISPLAY_NAME)]


def _balance_tracker():
    return [Tracker('eco', PERSON_BALANCE_ECO_PARAMETER, PERSON_BALANCE_ECO_RUN_ID, PERSON_BALANCE_ECO_DISPLAY_NAME)]


def _balance_base_tracker():
    return [Tracker('eco', PERSON_BALANCE_BASE_ECO_PARAMETER, PERSON_BALANCE_BASE_ECO_RUN_ID, PERSON_BALANCE_BASE_ECO_DISPLAY_NAME)]


def _rt_tracker():
    return [Tracker('eco', PERSON_RT_ECO_PARAMETER, PERSON_RT_ECO_RUN_ID, PERSON_RT_ECO_DISPLAY_NAME)]


def _rt_noproj_tracker():
    return [Tracker('eco', PERSON_RT_NOPROJ_ECO_PARAMETER, PERSON_RT_NOPROJ_ECO_RUN_ID, PERSON_RT_NOPROJ_ECO_DISPLAY_NAME)]


def _proj_rt_tracker():
    return [Tracker('eco', PERSON_PROJ_RT_ECO_PARAMETER, PERSON_PROJ_RT_ECO_RUN_ID, PERSON_PROJ_RT_ECO_DISPLAY_NAME)]


def _proj_rt_plus_tracker():
    return [Tracker('eco', PERSON_PROJ_RT_PLUS_ECO_PARAMETER, PERSON_PROJ_RT_PLUS_ECO_RUN_ID, PERSON_PROJ_RT_PLUS_ECO_DISPLAY_NAME)]


def _proj_rt_cg50_tracker():
    return [Tracker('eco', PERSON_PROJ_RT_CG50_ECO_PARAMETER, PERSON_PROJ_RT_CG50_ECO_RUN_ID, PERSON_PROJ_RT_CG50_ECO_DISPLAY_NAME)]


def _proj_rt_fb54_tracker():
    return [Tracker('eco', PERSON_PROJ_RT_FB54_ECO_PARAMETER, PERSON_PROJ_RT_FB54_ECO_RUN_ID, PERSON_PROJ_RT_FB54_ECO_DISPLAY_NAME)]


def _lite_proj_rt_tracker():
    return [Tracker('eco', PERSON_LITE_PROJ_RT_ECO_PARAMETER, PERSON_LITE_PROJ_RT_ECO_RUN_ID, PERSON_LITE_PROJ_RT_ECO_DISPLAY_NAME)]


def eco_person_otb():
    dataset = get_dataset('otb')
    dataset = dataset[[idx for idx, seq in enumerate(dataset) if seq.name in OTB_PERSON_SEQUENCES]]
    return _tracker(), dataset


def eco_person_lasot():
    dataset = get_dataset('lasot')
    dataset = dataset[[idx for idx, seq in enumerate(dataset) if seq.name.startswith('person-')]]
    return _tracker(), dataset


def eco_person_fast_otb():
    dataset = get_dataset('otb')
    dataset = dataset[[idx for idx, seq in enumerate(dataset) if seq.name in OTB_PERSON_SEQUENCES]]
    return _fast_tracker(), dataset


def eco_person_fast_lasot():
    dataset = get_dataset('lasot')
    dataset = dataset[[idx for idx, seq in enumerate(dataset) if seq.name.startswith('person-')]]
    return _fast_tracker(), dataset


def eco_person_balance_otb():
    dataset = get_dataset('otb')
    dataset = dataset[[idx for idx, seq in enumerate(dataset) if seq.name in OTB_PERSON_SEQUENCES]]
    return _balance_tracker(), dataset


def eco_person_balance_lasot():
    dataset = get_dataset('lasot')
    dataset = dataset[[idx for idx, seq in enumerate(dataset) if seq.name.startswith('person-')]]
    return _balance_tracker(), dataset


def eco_person_balance_base_otb():
    dataset = get_dataset('otb')
    dataset = dataset[[idx for idx, seq in enumerate(dataset) if seq.name in OTB_PERSON_SEQUENCES]]
    return _balance_base_tracker(), dataset


def eco_person_balance_base_lasot():
    dataset = get_dataset('lasot')
    dataset = dataset[[idx for idx, seq in enumerate(dataset) if seq.name.startswith('person-')]]
    return _balance_base_tracker(), dataset


def eco_person_rt_otb():
    dataset = get_dataset('otb')
    dataset = dataset[[idx for idx, seq in enumerate(dataset) if seq.name in OTB_PERSON_SEQUENCES]]
    return _rt_tracker(), dataset


def eco_person_rt_lasot():
    dataset = get_dataset('lasot')
    dataset = dataset[[idx for idx, seq in enumerate(dataset) if seq.name.startswith('person-')]]
    return _rt_tracker(), dataset


def eco_person_rt_noproj_otb():
    dataset = get_dataset('otb')
    dataset = dataset[[idx for idx, seq in enumerate(dataset) if seq.name in OTB_PERSON_SEQUENCES]]
    return _rt_noproj_tracker(), dataset


def eco_person_rt_noproj_lasot():
    dataset = get_dataset('lasot')
    dataset = dataset[[idx for idx, seq in enumerate(dataset) if seq.name.startswith('person-')]]
    return _rt_noproj_tracker(), dataset


def eco_person_proj_rt_otb():
    dataset = get_dataset('otb')
    dataset = dataset[[idx for idx, seq in enumerate(dataset) if seq.name in OTB_PERSON_SEQUENCES]]
    return _proj_rt_tracker(), dataset


def eco_person_proj_rt_lasot():
    dataset = get_dataset('lasot')
    dataset = dataset[[idx for idx, seq in enumerate(dataset) if seq.name.startswith('person-')]]
    return _proj_rt_tracker(), dataset


def eco_person_proj_rt_plus_otb():
    dataset = get_dataset('otb')
    dataset = dataset[[idx for idx, seq in enumerate(dataset) if seq.name in OTB_PERSON_SEQUENCES]]
    return _proj_rt_plus_tracker(), dataset


def eco_person_proj_rt_plus_lasot():
    dataset = get_dataset('lasot')
    dataset = dataset[[idx for idx, seq in enumerate(dataset) if seq.name.startswith('person-')]]
    return _proj_rt_plus_tracker(), dataset


def eco_person_proj_rt_cg50_otb():
    dataset = get_dataset('otb')
    dataset = dataset[[idx for idx, seq in enumerate(dataset) if seq.name in OTB_PERSON_SEQUENCES]]
    return _proj_rt_cg50_tracker(), dataset


def eco_person_proj_rt_cg50_lasot():
    dataset = get_dataset('lasot')
    dataset = dataset[[idx for idx, seq in enumerate(dataset) if seq.name.startswith('person-')]]
    return _proj_rt_cg50_tracker(), dataset


def eco_person_proj_rt_fb54_otb():
    dataset = get_dataset('otb')
    dataset = dataset[[idx for idx, seq in enumerate(dataset) if seq.name in OTB_PERSON_SEQUENCES]]
    return _proj_rt_fb54_tracker(), dataset


def eco_person_proj_rt_fb54_lasot():
    dataset = get_dataset('lasot')
    dataset = dataset[[idx for idx, seq in enumerate(dataset) if seq.name.startswith('person-')]]
    return _proj_rt_fb54_tracker(), dataset


def eco_person_lite_proj_rt_otb():
    dataset = get_dataset('otb')
    dataset = dataset[[idx for idx, seq in enumerate(dataset) if seq.name in OTB_PERSON_SEQUENCES]]
    return _lite_proj_rt_tracker(), dataset


def eco_person_lite_proj_rt_lasot():
    dataset = get_dataset('lasot')
    dataset = dataset[[idx for idx, seq in enumerate(dataset) if seq.name.startswith('person-')]]
    return _lite_proj_rt_tracker(), dataset
