import numpy as np

def create_logisticloss_label(label_size,rPos,rNeg):
    label_side = label_size[0]
    logloss_label = np.zeros((label_side,label_side))
    label_origin = [np.ceil(label_side/2),np.ceil(label_side/2)]
    for i in range(label_side):
        for j in range(label_side):
            dist_from_origin = ((i - label_origin[0])**2 + (j - label_origin[1])**2) ** 0.5
            if dist_from_origin <= rPos:
                logloss_label[i,j] = 1
            else:
                if dist_from_origin <= rNeg:
                    logloss_label[i,j] = 0
                else:
                    logloss_label[i,j] = -1
    return logloss_label

def create_labels(fixedLabelSize,rPos,rNeg):
    half = np.floor(fixedLabelSize[0] / 2) + 1
    fixedLabel = create_logisticloss_label(fixedLabelSize,rPos,rNeg)
    instanceWeight = np.ones(fixedLabel.shape)
    sumP = len([val for val in fixedLabel.flatten() if val == 1])
    sumN = len([val for val in fixedLabel.flatten() if val == -1])
    instanceWeight[fixedLabel == 1] = 0.5*instanceWeight[fixedLabel == 1] / sumP
    instanceWeight[fixedLabel == -1] = 0.5*instanceWeight[fixedLabel == -1] / sumN
    return fixedLabel,instanceWeight


