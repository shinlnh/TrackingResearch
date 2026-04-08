import sys
import numpy as np
import os
import time
import matplotlib.pyplot as plt
from utils import image_info
import cv2
from numpy.random import *
from pylab import *
from PIL import Image
import tempfile
import random
import re
from region_to_bbox import region_to_bbox
import tensorflow as tf
from siamese_net import SiameseNet

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def getSubWinTracking(img, pos, modelSz, originalSz, avgChans):
    if originalSz is None:
        originalSz = modelSz

    sz = originalSz
    im_sz = img.shape
    # make sure the size is not too small
    assert min(im_sz[:2]) > 2, "the size is too small"
    c = (np.array(sz) + 1) / 2

    # check out-of-bounds coordinates, and set them to black
    context_xmin = round(pos[1] - c[1])
    context_xmax = context_xmin + sz[1] - 1
    context_ymin = round(pos[0] - c[0])
    context_ymax = context_ymin + sz[0] - 1
    left_pad = max(0, int(-context_xmin))
    top_pad = max(0, int(-context_ymin))
    right_pad = max(0, int(context_xmax - im_sz[1] + 1))
    bottom_pad = max(0, int(context_ymax - im_sz[0] + 1))

    context_xmin = int(context_xmin + left_pad)
    context_xmax = int(context_xmax + left_pad)
    context_ymin = int(context_ymin + top_pad)
    context_ymax = int(context_ymax + top_pad)

    if top_pad or left_pad or bottom_pad or right_pad:
        r = np.pad(img[:, :, 0], ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant',
                   constant_values=avgChans[0])
        g = np.pad(img[:, :, 1], ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant',
                   constant_values=avgChans[1])
        b = np.pad(img[:, :, 2], ((top_pad, bottom_pad), (left_pad, right_pad)), mode='constant',
                   constant_values=avgChans[2])
        r = np.expand_dims(r, 2)
        g = np.expand_dims(g, 2)
        b = np.expand_dims(b, 2)

        # h, w = r.shape
        # r1 = np.zeros([h, w, 1], dtype=np.float32)
        # r1[:, :, 0] = r
        # g1 = np.zeros([h, w, 1], dtype=np.float32)
        # g1[:, :, 0] = g
        # b1 = np.zeros([h, w, 1], dtype=np.float32)
        # b1[:, :, 0] = b

        img = np.concatenate((r, g, b ), axis=2)

    im_patch_original = img[context_ymin:context_ymax + 1, context_xmin:context_xmax + 1, :]
    if not np.array_equal(modelSz, originalSz):
        im_patch = cv2.resize(im_patch_original, modelSz)
        # im_patch_original = im_patch_original/255.0
        # im_patch = transform.resize(im_patch_original, modelSz)*255.0
        # im = Image.fromarray(im_patch_original.astype(np.float))
        # im = im.resize(modelSz)
        # im_patch = np.array(im).astype(np.float32)
    else:
        im_patch = im_patch_original

    return im_patch, im_patch_original

def _update_target_position(pos_x, pos_y, score, final_score_sz, tot_stride, search_sz, response_up, x_sz):
    # find location of score maximizer
    p = np.asarray(np.unravel_index(np.argmax(score), np.shape(score)))
    # displacement from the center in search area final representation ...
    center = float(final_score_sz - 1) / 2
    disp_in_area = p - center
    # displacement from the center in instance crop
    disp_in_xcrop = disp_in_area * float(tot_stride) / response_up
    # displacement from the center in instance crop (in frame coordinates)
    disp_in_frame = disp_in_xcrop *  x_sz / search_sz
    # *position* within frame in frame coordinates
    pos_y, pos_x = pos_y + disp_in_frame[0], pos_x + disp_in_frame[1]
    return pos_x, pos_y

def trackerEval(score, sx, targetPosition, window, hp,design):
    # responseMaps = np.transpose(score[:, :, :, 0], [1, 2, 0])
    responseMaps = score[:,:,:,0]
    upsz = design['score_sz']*hp['response_up']
    # responseMapsUp = np.zeros([opts['scoreSize']*opts['responseUp'], opts['scoreSize']*opts['responseUp'], opts['numScale']])
    responseMapsUP = []

    if hp['scale_num'] > 1:
        currentScaleID = int(hp['scale_num']/2)
        bestScale = currentScaleID
        bestPeak = -float('Inf')

        for s in range(hp['scale_num']):
            if hp['response_up'] > 1:
                responseMapsUP.append(cv2.resize(responseMaps[s, :, :], (upsz, upsz), interpolation=cv2.INTER_CUBIC))
            else:
                responseMapsUP.append(responseMaps[s, :, :])

            thisResponse = responseMapsUP[-1]

            if s != currentScaleID:
                thisResponse = thisResponse*hp['scale_penalty']

            thisPeak = np.max(thisResponse)
            if thisPeak > bestPeak:
                bestPeak = thisPeak
                bestScale = s

        responseMap = responseMapsUP[bestScale]
    else:
        responseMap = cv2.resize(responseMaps[0, :, :], (upsz, upsz), interpolation=cv2.INTER_CUBIC)
        bestScale = 0

    responseMap = responseMap - np.min(responseMap)
    responseMap = responseMap/np.sum(responseMap)

    responseMap = (1-hp['window_influence'])*responseMap+hp['window_influence']*window
    rMax, cMax = np.unravel_index(responseMap.argmax(), responseMap.shape)
    pCorr = np.array((rMax, cMax))
    dispInstanceFinal = pCorr-int(upsz/2)
    dispInstanceInput = dispInstanceFinal*design['tot_stride']/hp['response_up']
    dispInstanceFrame = dispInstanceInput*sx/design['search_sz']
    newTargetPosition = targetPosition+dispInstanceFrame
    # print(bestScale)

    return newTargetPosition, bestScale

def makeScalePyramid(im, targetPosition, in_side_scaled, out_side, avgChans, numScale):
    """
    computes a pyramid of re-scaled copies of the target (centered on TARGETPOSITION)
    and resizes them to OUT_SIDE. If crops exceed image boundaries they are padded with AVGCHANS.
    """
    in_side_scaled = np.round(in_side_scaled)
    max_target_side = int(round(in_side_scaled[-1]))
    min_target_side = int(round(in_side_scaled[0]))
    beta = out_side / float(min_target_side)
    # size_in_search_area = beta * size_in_image
    # e.g. out_side = beta * min_target_side
    search_side = int(round(beta * max_target_side))
    search_region, _ = getSubWinTracking(im, targetPosition, (search_side, search_side),
                                              (max_target_side, max_target_side), avgChans)

    assert round(beta * min_target_side) == int(out_side)

    tmp_list = []
    tmp_pos = ((search_side - 1) / 2., (search_side - 1) / 2.)
    for s in range(numScale):
        target_side = round(beta * in_side_scaled[s])
        tmp_region, _ = getSubWinTracking(search_region, tmp_pos, (out_side, out_side), (target_side, target_side),
                                               avgChans)
        tmp_list.append(tmp_region)

    pyramid = np.stack(tmp_list)

    return pyramid

def _init_video(video,sequences_dir):
    root_dataset = sequences_dir + '/' + re.findall('[a-zA-Z]+', video)[0] + '/'
    video += '/'
    video_folder = os.path.join(root_dataset, video,'img')
    frame_name_list = [f for f in os.listdir(video_folder) if f.endswith(".jpg")]
    frame_name_list = [os.path.join(root_dataset, video,'img', '') + s for s in frame_name_list]
    frame_name_list.sort()
    with Image.open(frame_name_list[0]) as img:
        frame_sz = np.asarray(img.size)
        frame_sz[1], frame_sz[0] = frame_sz[0], frame_sz[1]

    # read the initialization from ground truth
    gt_file = os.path.join(root_dataset,video, 'groundtruth.txt')
    gt = np.genfromtxt(gt_file, delimiter=',')
    # if len(gt.shape) < 2:
    #     gt = np.genfromtxt(gt_file)
    n_frames = len(frame_name_list)
    # if n_frames > len(gt):
    #     frame_name_list = frame_name_list[:len(gt)]

    return gt, frame_name_list, frame_sz, n_frames

def getOpts(opts):
    print("config opts...")

    opts['numScale'] = 3
    opts['scaleStep'] = 1.0375
    opts['scalePenalty'] = 0.9745
    # opts['scalePenalty'] = 1/0.9745
    opts['scaleLr'] = 0.59
    opts['responseUp'] = 16
    opts['windowing'] = 'cosine'
    opts['wInfluence'] = 0.176
    opts['exemplarSize'] = 127
    opts['instanceSize'] = 255
    opts['scoreSize'] = 17
    opts['totalStride'] = 8
    opts['contextAmount'] = 0.5
    opts['trainWeightDecay'] = 5e-04
    opts['stddev'] = 0.03
    opts['subMean'] = False

    opts['video'] = 'vot15_bag'
    opts['modelPath'] = './models/'
    opts['modelName'] = opts['modelPath']+"model_tf.ckpt"
    opts['summaryFile'] = './data_track/'+opts['video']+'_20170518'

    return opts

if __name__ == '__main__':
    sequences_dir = '/home/lab/VideoNetTrackerEvaluation/VideoNetBenchmarkTest'
    titles = []
    category = os.listdir(sequences_dir)
    for i in range(len(category)):
        titles += os.listdir(sequences_dir + '/' + category[i])
    titles.sort()
    hp = {}
    hp['scale_min'] = 0.2
    hp['window_influence'] = 0.25
    hp['z_lr'] = 0.01
    hp['scale_max'] = 5
    hp['scale_step'] = 1.04
    hp['scale_num'] = 3
    hp['scale_penalty'] = 0.97
    hp['response_up'] = 8
    hp['scale_lr'] = 0.59

    evaluation = {}
    evaluation['start_frame'] = 0
    evaluation['n_subseq'] = 1
    evaluation['stop_on_failure'] = 0
    evaluation['dist_threshold'] = 20

    design = {}
    design['exemplar_sz'] = 127
    design['search_sz'] = 255
    design['tot_stride'] = 4
    design['context'] = 0.5
    design['pad_with_image_mean'] = True
    design['windowing'] = 'cosine_sum'
    design['score_sz'] = 33
    design['trainBatchSize'] = 8


    opts = {}
    opts = getOpts(opts)

    exemplarOp = tf.placeholder(tf.float32, [1, design['exemplar_sz'], design['exemplar_sz'], 3])
    instanceOp = tf.placeholder(tf.float32, [hp['scale_num'], design['search_sz'], design['search_sz'], 3])
    exemplarOpBak = tf.placeholder(tf.float32, [design['trainBatchSize'], design['exemplar_sz'], design['exemplar_sz'], 3])
    instanceOpBak = tf.placeholder(tf.float32, [design['trainBatchSize'], design['search_sz'], design['search_sz'], 3])
    isTrainingOp = tf.convert_to_tensor(False, dtype='bool', name='is_training')

    sn = SiameseNet()
    scoreOpBak = sn.buildTrainNetwork(exemplarOpBak, instanceOpBak, opts, isTraining=False)
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter('./summaryFile/summary')
    sess = tf.Session()
    saver.restore(sess, './ckpt/ckpt/model_epoch49.ckpt')

    zFeatOp = sn.buildExemplarSubNetwork(exemplarOp, opts, isTrainingOp)

    nv = len(titles)
    speed = np.zeros(nv * evaluation['n_subseq'])
    precisions = np.zeros(nv * evaluation['n_subseq'])
    precisions_auc = np.zeros(nv * evaluation['n_subseq'])
    ious = np.zeros(nv * evaluation['n_subseq'])
    lengths = np.zeros(nv * evaluation['n_subseq'])
    for i in range(nv):
        gt, frame_name_list, frame_sz, n_frames = _init_video(titles[i],sequences_dir)
        starts = np.rint(np.linspace(0, n_frames - 1, evaluation['n_subseq'] + 1))
        starts = starts[0:evaluation['n_subseq']]
        final_score_sz = hp['response_up'] * (design['score_sz'] - 1) + 1
        for j in range(evaluation['n_subseq']):
            idx = i * evaluation['n_subseq'] + j
            start_frame = int(starts[j])
            gt_ = gt
            frame_name_list_ = frame_name_list[start_frame:]
            pos_x, pos_y, target_w, target_h = region_to_bbox(gt_)
            '''----------------------tracking single video------------------------'''
            num_frames = np.size(frame_name_list)
            scale_factors = hp['scale_step'] ** np.linspace(-np.ceil(hp['scale_num'] / 2), np.ceil(hp['scale_num'] / 2),
                                                            hp['scale_num'])
            hann_1d = np.expand_dims(np.hanning(final_score_sz), axis=0)
            penalty = np.transpose(hann_1d) * hann_1d
            penalty = penalty / np.sum(penalty)
            context = design['context'] * (target_w + target_h)
            z_sz = np.sqrt(np.prod((target_w + context) * (target_h + context)))
            scalez = design['exemplar_sz'] / z_sz
            im = cv2.imread(frame_name_list[start_frame], cv2.IMREAD_COLOR)
            avgChans = np.mean(im, axis=(0, 1))

            zCrop, _ = getSubWinTracking(im, [pos_y, pos_x], (design['exemplar_sz'], design['exemplar_sz']),
                                         (np.around(z_sz), np.around(z_sz)), avgChans)
            dSearch = (design['search_sz'] - design['exemplar_sz']) / 2
            pad = dSearch / scalez
            sx = z_sz + 2 * pad

            minSx = 0.2 * sx
            maxSx = 5.0 * sx
            winSz = design['score_sz'] * hp['response_up']

            hann = np.hanning(winSz).reshape(winSz, 1)
            window = hann.dot(hann.T)

            window = window / np.sum(window)
            scales = np.array([hp['scale_step'] ** k for k in
                               range(int(np.ceil(hp['scale_num'] / 2.0) - hp['scale_num']),
                                     int(np.floor(hp['scale_num'] / 2.0) + 1))])
            zCrop = np.expand_dims(zCrop, axis=0)
            zFeat = sess.run(zFeatOp, feed_dict={exemplarOp: zCrop.astype('float')})
            zFeat = np.transpose(zFeat, [1, 2, 3, 0])
            zFeatConstantOp = tf.constant(zFeat, dtype=tf.float32)
            scoreOp = sn.buildInferenceNetwork(instanceOp, zFeatConstantOp, opts, isTrainingOp)
            writer.add_graph(sess.graph)

            bBoxes = np.zeros([len(frame_name_list) - start_frame, 4])
            bBoxes[0, :] = pos_x - target_w / 2, pos_y - target_h / 2, target_w, target_h
            targetSize = np.array([target_h, target_w])
            t_start = time.time()
            for im_id in range(start_frame + 1, len(frame_name_list)):
                im = cv2.imread(frame_name_list[im_id], cv2.IMREAD_COLOR)
                scaledInstance = sx * scales
                scaledTarget = np.array([targetSize * scale_i for k, scale_i in enumerate(scales)])
                xCrops = makeScalePyramid(im, [pos_y, pos_x], scaledInstance, design['search_sz'], avgChans,
                                          hp['scale_num'])
                score = sess.run(scoreOp, feed_dict={instanceOp: xCrops})
                newTargetPosition, newScale = trackerEval(score, round(sx), [pos_y, pos_x], window, hp, design)
                targetPosition = newTargetPosition
                sx = max(minSx, min(maxSx, (1 - hp['scale_lr']) * sx + hp['scale_lr'] * scaledInstance[newScale]))

                targetSize = (1 - hp['scale_lr']) * targetSize + hp['scale_lr'] * scaledTarget[newScale]

                rectPosition = targetPosition - targetSize / 2.
                tl = tuple(np.round(rectPosition).astype(int)[::-1])
                br = tuple(np.round(rectPosition + targetSize).astype(int)[::-1])
                imDraw = im.astype(np.uint8)
                # cv2.rectangle(imDraw, tl, br, (0, 255, 255), thickness=3)
                # plt.imshow(imDraw)
                # cv2.waitKey(1)
                # plt.show()
                # plt.pause(0.001)
                bBoxes[im_id-start_frame, :] = tl[0], tl[1], br[0] - tl[0], br[1] - tl[1]
                pos_y = tl[1] + (br[1] - tl[1])/2
                pos_x = tl[0] + (br[0] - tl[0])/2

            t_elapsed = time.time() - t_start
            speed_i = num_frames / t_elapsed
            print titles[i] + ': ' + str(speed_i)
            np.savetxt(titles[i] + '_results.txt', bBoxes.astype(np.int32), fmt='%i', delimiter=",")
            speed[idx] = speed_i

