from pytracking.evaluation import Tracker, get_dataset, trackerlist


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
    trackers = [Tracker('eco', 'verified_otb936', 936, 'MyTrackerECO')]
    dataset = get_dataset('otb')
    return trackers, dataset


def eco_verified_otb936_lasot():
    trackers = [Tracker('eco', 'verified_otb936', 936, 'MyTrackerECO')]
    dataset = get_dataset('lasot')
    return trackers, dataset


def eco_verified_otb936_lasot_first20():
    trackers = [Tracker('eco', 'verified_otb936', 936, 'MyTrackerECO')]
    dataset = get_dataset('lasot')
    return trackers, dataset[:20]


def eco_verified_otb936_lasot_headtail40():
    trackers = [Tracker('eco', 'verified_otb936', 936, 'MyTrackerECO')]
    dataset = get_dataset('lasot')
    return trackers, dataset[:20] + dataset[-20:]
