from pathlib import Path

from pytracking.evaluation import Tracker, get_dataset, trackerlist
from pytracking.evaluation.lasotdataset import LaSOTDataset

MAIN_ECO_PARAMETER = 'verified_otb936_main'
MAIN_ECO_RUN_ID = 953
MAIN_ECO_DISPLAY_NAME = 'MyTrackerECO-Main'
RUN_UPDATE_ECO_PARAMETER = 'verified_otb936_run_update'
RUN_UPDATE_ECO_RUN_ID = 954
RUN_UPDATE_ECO_DISPLAY_NAME = 'MyTrackerECO-RunUpdate'
LASOT_HEADTAIL40_SEQUENCE_FILE = Path(__file__).resolve().parents[3] / 'jetson' / 'lasot_headtail40_sequences.txt'


def _load_lasot_subset(sequence_file=LASOT_HEADTAIL40_SEQUENCE_FILE, limit=None):
    sequence_names = [line.strip() for line in sequence_file.read_text(encoding='utf-8').splitlines() if line.strip()]
    if limit is not None:
        sequence_names = sequence_names[:limit]
    return LaSOTDataset(sequence_list=sequence_names).get_sequence_list()


def atom_nfs_uav():
    # Run three runs of ATOM on NFS and UAV datasets
    trackers = trackerlist('atom', 'default', range(3))

    dataset = get_dataset('nfs', 'uav')
    return trackers, dataset


def uav_test():
    # Run DiMP18, ATOM and ECO on the UAV dataset
    trackers = trackerlist('dimp', 'dimp18', range(1)) + \
               trackerlist('atom', 'default', range(1)) + \
               trackerlist('eco', 'default', range(1))

    dataset = get_dataset('uav')
    return trackers, dataset


def eco_verified_otb936_otb():
    trackers = [Tracker('eco', MAIN_ECO_PARAMETER, MAIN_ECO_RUN_ID, MAIN_ECO_DISPLAY_NAME)]
    dataset = get_dataset('otb')
    return trackers, dataset


def eco_verified_otb936_otb_easy3():
    trackers = [Tracker('eco', MAIN_ECO_PARAMETER, MAIN_ECO_RUN_ID, MAIN_ECO_DISPLAY_NAME)]

    from pytracking.evaluation.otbdataset import OTBDataset

    wanted = {'Deer', 'DragonBaby', 'Ironman'}
    dataset_obj = OTBDataset()
    dataset_obj.sequence_info_list = [s for s in dataset_obj.sequence_info_list if s['name'] in wanted]
    dataset = dataset_obj.get_sequence_list()
    return trackers, dataset


def eco_verified_otb936_lasot():
    trackers = [Tracker('eco', MAIN_ECO_PARAMETER, MAIN_ECO_RUN_ID, MAIN_ECO_DISPLAY_NAME)]
    dataset = get_dataset('lasot')
    return trackers, dataset


def eco_verified_otb936_lasot_first20():
    trackers = [Tracker('eco', MAIN_ECO_PARAMETER, MAIN_ECO_RUN_ID, MAIN_ECO_DISPLAY_NAME)]
    dataset = _load_lasot_subset(limit=20)
    return trackers, dataset


def eco_verified_otb936_lasot_headtail40():
    trackers = [Tracker('eco', MAIN_ECO_PARAMETER, MAIN_ECO_RUN_ID, MAIN_ECO_DISPLAY_NAME)]
    dataset = _load_lasot_subset()
    return trackers, dataset


def eco_verified_otb936_run_update_otb():
    trackers = [Tracker('eco', RUN_UPDATE_ECO_PARAMETER, RUN_UPDATE_ECO_RUN_ID, RUN_UPDATE_ECO_DISPLAY_NAME)]
    dataset = get_dataset('otb')
    return trackers, dataset


def eco_verified_otb936_run_update_lasot():
    trackers = [Tracker('eco', RUN_UPDATE_ECO_PARAMETER, RUN_UPDATE_ECO_RUN_ID, RUN_UPDATE_ECO_DISPLAY_NAME)]
    dataset = get_dataset('lasot')
    return trackers, dataset


def eco_verified_otb936_run_update_lasot_first20():
    trackers = [Tracker('eco', RUN_UPDATE_ECO_PARAMETER, RUN_UPDATE_ECO_RUN_ID, RUN_UPDATE_ECO_DISPLAY_NAME)]
    dataset = _load_lasot_subset(limit=20)
    return trackers, dataset


def eco_verified_otb936_run_update_lasot_headtail40():
    trackers = [Tracker('eco', RUN_UPDATE_ECO_PARAMETER, RUN_UPDATE_ECO_RUN_ID, RUN_UPDATE_ECO_DISPLAY_NAME)]
    dataset = _load_lasot_subset()
    return trackers, dataset
