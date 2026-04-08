import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.measure import block_reduce
import os

def padarray(arr, pad):
    if len(arr.shape) == 3:
        padded = np.zeros((arr.shape[0]+2*pad,arr.shape[1]+2*pad,arr.shape[2]))
        padded = padded.astype('uint8')
        padded[pad:(pad+arr.shape[0]),pad:(pad+arr.shape[1]),:] = arr
    else:
        padded = np.zeros((arr.shape[0]+2*pad,arr.shape[1]+2*pad))
        padded = padded.astype('uint8')
        padded[pad:(pad+arr.shape[0]),pad:(pad+arr.shape[1])] = arr
    return padded


def ext_roi(im, GT, l_off, roi_size, r_w_scale,shuffle=False):
    im_h = im.shape[0]
    im_w = im.shape[1]
    dia = (GT[2] ** 2 + GT[3] ** 2) ** 0.5
    win_w = GT[2]
    win_h = GT[3]
    win_lt_x = GT[0]
    win_lt_y = GT[1]
    win_cx = round(win_lt_x + win_w / 2.0 + l_off[0])
    win_cy = round(win_lt_y + win_h / 2.0 + l_off[1])
    roi_w = r_w_scale[0] * win_w
    roi_h = r_w_scale[1] * win_h
    #scale = np.random.rand(1)[0]
    #roi_w = roi_w - scale * r_w_scale[0] * win_w / 2.0
    #roi_h = roi_h - scale * r_w_scale[1] * win_h / 2.0
    x1 = int(win_cx - roi_w / 2.0)
    y1 = int(win_cy - roi_h / 2.0)
    x2 = int(win_cx + roi_w / 2.0)
    y2 = int(win_cy + roi_h / 2.0)

    if shuffle == True:
        x_remain = np.min([GT[0] - x1, x2 - (GT[0] + GT[2])])
        y_remain = np.min([GT[1] - y1, y2 - (GT[1] + GT[3])])
        if np.random.rand(1)[0] > 0.5:
            x_off = int(np.round(np.random.rand(1)[0] * x_remain))
        else:
            x_off = -int(np.round(np.random.rand(1)[0] * x_remain))
        if np.random.rand(1)[0] > 0.5:
            y_off = int(np.round(np.random.rand(1)[0] * y_remain))
        else:
            y_off = -int(np.round(np.random.rand(1)[0] * y_remain))
    else:
        x_off = l_off[0]
        y_off = l_off[1]
    x1 = x1 #+ x_off
    y1 = y1 #+ y_off
    x2 = x2 #+ x_off
    y2 = y2 #+ y_off

    # im = im.astype('float32')
    clip = min([x1, y1, im_h - y2, im_w - x2])
    pad = 0
    if clip <= 0:
        pad = int(abs(clip) + 1)
        im = padarray(im, pad)
        x1 = x1 + pad
        x2 = x2 + pad
        y1 = y1 + pad
        y2 = y2 + pad
    if len(im.shape) == 3:
        roi = cv2.resize(im[int(y1):int(y2), int(x1):int(x2), :], (roi_size, roi_size),interpolation = cv2.INTER_AREA)
    else:
        roi = cv2.resize(im[int(y1):int(y2), int(x1):int(x2)], (roi_size, roi_size),interpolation = cv2.INTER_AREA)

    return roi, roi_w, roi_h, [x_off,y_off]

def GetPartMap(im_sz, fea_sz, roi_size, location,part_location, l_off, s,radius):
    w_im = location[2]
    h_im = location[3]
    #dia = ((w_im ** 2 + h_im ** 2) ** 0.5)
    xstart = location[0] + location[2]/2.0 - radius + l_off[0]
    ystart = location[1] + location[3]/2.0 - radius + l_off[1]

    x_part = part_location[0]
    y_part = part_location[1]

    x_part = x_part + location[0]
    y_part = y_part + location[1]
    x_part = (x_part - xstart)/(2*radius) * float(fea_sz[0])
    y_part = (y_part - ystart)/(2*radius) * float(fea_sz[0])

    x_part = int(np.round(x_part))
    y_part = int(np.round(y_part))
    if y_part > 63:
        y_part = 63
    if y_part < 0:
        y_part = 0
    if x_part > 63:
        x_part = 63
    if x_part < 0:
        x_part = 0
    map_get = np.zeros((fea_sz[0],fea_sz[1]))
    map_get[y_part,x_part] = 1.0
    if x_part - 1 > 0:
        if y_part - 1 > 0:
            map_get[y_part-1,x_part-1] = 0.1
        map_get[y_part,x_part-1] = 0.3
        if y_part + 1 <= fea_sz[1]-1:
            map_get[y_part+1,x_part-1] = 0.1
    if y_part-1>0:
        map_get[y_part-1,x_part] = 0.3
    if y_part+1<=fea_sz[0]-1:
        map_get[y_part+1,x_part] = 0.3
    if x_part+1 <= fea_sz[1]-1:
        if y_part-1 >= 0:
            map_get[y_part-1,x_part+1] = 0.1
        map_get[y_part,x_part+1] = 0.3
        if y_part +1 <= fea_sz[0]-1:
            map_get[y_part+1,x_part+1] = 0.1

    return map_get,x_part,y_part

def GetPartMap1(im_sz, fea_sz, location,part_location):
    x_part = part_location[0]
    y_part = part_location[1]

    x_part = x_part + location[0]
    y_part = y_part + location[1]

    x_part = x_part / float(im_sz[1]) * 64.0
    y_part = y_part / float(im_sz[0]) * 64.0

    x_part = int(np.round(x_part))
    y_part = int(np.round(y_part))
    if y_part > 63:
        y_part = 63
    if y_part < 0:
        y_part = 0
    if x_part > 63:
        x_part = 63
    if x_part < 0:
        x_part = 0
    map_get = np.zeros((fea_sz[0],fea_sz[1]))
    map_get[y_part,x_part] = 1.0
    if x_part - 1 > 0:
        if y_part - 1 > 0:
            map_get[y_part-1,x_part-1] = 0.1
        map_get[y_part,x_part-1] = 0.3
        if y_part + 1 <= fea_sz[1]-1:
            map_get[y_part+1,x_part-1] = 0.1
    if y_part-1>0:
        map_get[y_part-1,x_part] = 0.3
    if y_part+1<=fea_sz[0]-1:
        map_get[y_part+1,x_part] = 0.3
    if x_part+1 <= fea_sz[1]-1:
        if y_part-1 >= 0:
            map_get[y_part-1,x_part+1] = 0.1
        map_get[y_part,x_part+1] = 0.3
        if y_part +1 <= fea_sz[0]-1:
            map_get[y_part+1,x_part+1] = 0.1

    return map_get,x_part,y_part


def get9part1(im_sz, fea_sz, location, w_im, h_im):
    map_center, x_part_tmp, y_part_tmp = GetPartMap1(im_sz, fea_sz, location,[0,0])
    part_loc = np.asarray([x_part_tmp, y_part_tmp])
    part_loc = part_loc.reshape((1, 2))
    map_center = map_center.reshape((1, fea_sz[0], fea_sz[1]))
    map_center1, x_part_tmp, y_part_tmp = GetPartMap1(im_sz, fea_sz, location, [0, h_im])
    part_loc = np.concatenate((part_loc, np.asarray([x_part_tmp, y_part_tmp]).reshape((1, 2))), 0)
    map_center = np.concatenate((map_center, map_center1.reshape((1, fea_sz[0], fea_sz[1]))), axis=0)
    map_center1, x_part_tmp, y_part_tmp = GetPartMap1(im_sz, fea_sz, location, [w_im, 0])
    part_loc = np.concatenate((part_loc, np.asarray([x_part_tmp, y_part_tmp]).reshape((1, 2))), 0)
    map_center = np.concatenate((map_center, map_center1.reshape((1, fea_sz[0], fea_sz[1]))), axis=0)
    map_center1, x_part_tmp, y_part_tmp = GetPartMap1(im_sz, fea_sz, location, [w_im, h_im])
    part_loc = np.concatenate((part_loc, np.asarray([x_part_tmp, y_part_tmp]).reshape((1, 2))), 0)
    map_center = np.concatenate((map_center, map_center1.reshape((1, fea_sz[0], fea_sz[1]))), axis=0)
    map_center1, x_part_tmp, y_part_tmp = GetPartMap1(im_sz, fea_sz, location, [w_im / 2.0, h_im / 2.0])
    part_loc = np.concatenate((part_loc, np.asarray([x_part_tmp, y_part_tmp]).reshape((1, 2))), 0)
    map_center = np.concatenate((map_center, map_center1.reshape((1, fea_sz[0], fea_sz[1]))), axis=0)
    map_center1, x_part_tmp, y_part_tmp = GetPartMap1(im_sz, fea_sz, location,[w_im / 2.0, 0])
    part_loc = np.concatenate((part_loc, np.asarray([x_part_tmp, y_part_tmp]).reshape((1, 2))), 0)
    map_center = np.concatenate((map_center, map_center1.reshape((1, fea_sz[0], fea_sz[1]))), axis=0)
    map_center1, x_part_tmp, y_part_tmp = GetPartMap1(im_sz, fea_sz, location, [0, h_im / 2.0])
    part_loc = np.concatenate((part_loc, np.asarray([x_part_tmp, y_part_tmp]).reshape((1, 2))), 0)
    map_center = np.concatenate((map_center, map_center1.reshape((1, fea_sz[0], fea_sz[1]))), axis=0)
    map_center1, x_part_tmp, y_part_tmp = GetPartMap1(im_sz, fea_sz, location, [w_im / 2.0, h_im])
    part_loc = np.concatenate((part_loc, np.asarray([x_part_tmp, y_part_tmp]).reshape((1, 2))), 0)
    map_center = np.concatenate((map_center, map_center1.reshape((1, fea_sz[0], fea_sz[1]))), axis=0)
    map_center1, x_part_tmp, y_part_tmp = GetPartMap1(im_sz, fea_sz, location,[w_im, h_im / 2.0])
    part_loc = np.concatenate((part_loc, np.asarray([x_part_tmp, y_part_tmp]).reshape((1, 2))), 0)
    map_center = np.concatenate((map_center, map_center1.reshape((1, fea_sz[0], fea_sz[1]))), axis=0)

    return map_center, part_loc

def image_info(title):
    im_data=[]
    try:
        gt = open(r"/home/xiaobai/TB-50/%s/groundtruth_rect.txt" %title,'r')
    except IOError:
        gt = open(r"/home/xiaobai/TB-50/%s/groundtruth_rect.1.txt" %title,'r')
    except:
        gt = open(r"/home/xiaobai/TB-50/%s/groundtruth_rect.2.txt" %title,'r')
    lines = gt.readlines()
    for line in lines:
        line = line.strip('\n')
        try:
            row = [int(x) for x in line.split(',')]
        except ValueError:
            row = [int(x) for x in line.split('\t')]
        im_data.append(row)

    imageList = os.listdir('/home/xiaobai/TB-50/%s/img' %title)
    imageList.sort()
    return im_data, imageList

def get9part(im_sz, fea_sz, input_size, location, w_im, h_im, l_off, s1, ratio):
    map_center, x_part_tmp, y_part_tmp = GetPartMap(im_sz, fea_sz, input_size, location, [0, 0], l_off, s1, ratio)
    part_loc = np.asarray([x_part_tmp, y_part_tmp])
    part_loc = part_loc.reshape((1, 2))
    map_center = map_center.reshape((1, fea_sz[0], fea_sz[1]))
    map_center1, x_part_tmp, y_part_tmp = GetPartMap(im_sz, fea_sz, input_size, location, [0, h_im], l_off, s1,
                                                     ratio)
    part_loc = np.concatenate((part_loc, np.asarray([x_part_tmp, y_part_tmp]).reshape((1, 2))), 0)
    map_center = np.concatenate((map_center, map_center1.reshape((1, fea_sz[0], fea_sz[1]))), axis=0)
    map_center1, x_part_tmp, y_part_tmp = GetPartMap(im_sz, fea_sz, input_size, location, [w_im, 0], l_off, s1,
                                                     ratio)
    part_loc = np.concatenate((part_loc, np.asarray([x_part_tmp, y_part_tmp]).reshape((1, 2))), 0)
    map_center = np.concatenate((map_center, map_center1.reshape((1, fea_sz[0], fea_sz[1]))), axis=0)
    map_center1, x_part_tmp, y_part_tmp = GetPartMap(im_sz, fea_sz, input_size, location, [w_im, h_im], l_off, s1,
                                                     ratio)
    part_loc = np.concatenate((part_loc, np.asarray([x_part_tmp, y_part_tmp]).reshape((1, 2))), 0)
    map_center = np.concatenate((map_center, map_center1.reshape((1, fea_sz[0], fea_sz[1]))), axis=0)
    map_center1, x_part_tmp, y_part_tmp = GetPartMap(im_sz, fea_sz, input_size, location, [w_im / 2.0, h_im / 2.0],
                                                     l_off, s1, ratio)
    part_loc = np.concatenate((part_loc, np.asarray([x_part_tmp, y_part_tmp]).reshape((1, 2))), 0)
    map_center = np.concatenate((map_center, map_center1.reshape((1, fea_sz[0], fea_sz[1]))), axis=0)
    map_center1, x_part_tmp, y_part_tmp = GetPartMap(im_sz, fea_sz, input_size, location, [w_im / 2.0, 0], l_off,
                                                     s1, ratio)
    part_loc = np.concatenate((part_loc, np.asarray([x_part_tmp, y_part_tmp]).reshape((1, 2))), 0)
    map_center = np.concatenate((map_center, map_center1.reshape((1, fea_sz[0], fea_sz[1]))), axis=0)
    map_center1, x_part_tmp, y_part_tmp = GetPartMap(im_sz, fea_sz, input_size, location, [0, h_im / 2.0], l_off,
                                                     s1, ratio)
    part_loc = np.concatenate((part_loc, np.asarray([x_part_tmp, y_part_tmp]).reshape((1, 2))), 0)
    map_center = np.concatenate((map_center, map_center1.reshape((1, fea_sz[0], fea_sz[1]))), axis=0)
    map_center1, x_part_tmp, y_part_tmp = GetPartMap(im_sz, fea_sz, input_size, location, [w_im / 2.0, h_im],
                                                     l_off, s1, ratio)
    part_loc = np.concatenate((part_loc, np.asarray([x_part_tmp, y_part_tmp]).reshape((1, 2))), 0)
    map_center = np.concatenate((map_center, map_center1.reshape((1, fea_sz[0], fea_sz[1]))), axis=0)
    map_center1, x_part_tmp, y_part_tmp = GetPartMap(im_sz, fea_sz, input_size, location, [w_im, h_im / 2.0],
                                                     l_off, s1, ratio)
    part_loc = np.concatenate((part_loc, np.asarray([x_part_tmp, y_part_tmp]).reshape((1, 2))), 0)
    map_center = np.concatenate((map_center, map_center1.reshape((1, fea_sz[0], fea_sz[1]))), axis=0)

    return map_center, part_loc

def precision_plot(result,ground_truth):
    max_threshold = 50
    precisions = np.zeros((max_threshold,1))
    dd = (result[:,0] - ground_truth[:,0])**2 + (result[:,1] - ground_truth[:,1])**2
    distances = np.sqrt(dd)
    for p in range(max_threshold):
        count = 0
        for i in range(len(distances)):
            if distances[i] <= (p+1):
                count+=1
        precisions[p][0] = float(count)/len(distances)
    #plt.figure(22)
    x = range(0,max_threshold)
    #plt.plot(x,precisions[:,0],'r',label='accuracy')
    #plt.title("accuracy")
    #plt.ylabel("%")
    #plt.show()
    #x = range(0,max_threshold)
    #plt.plot(x, precisions[:,0])
    #plt.show()
    return precisions

def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

def GetMap(im_sz, fea_sz, roi_size, location, l_off, s):
    map_get = np.zeros((im_sz[0], im_sz[1]))
    scale = min(location[2],location[3])
    mask = makeGaussian(min(location[2],location[3]), fwhm = scale)
    mask = mask*255.0
    mask = mask.astype('uint8')
    mask = cv2.resize(mask,(int(location[2]),int(location[3])))

    x1 = location[0] + l_off[0]
    y1 = location[1] + l_off[1]
    x2 = x1+location[2] + l_off[0]
    y2 = y1+location[3] + l_off[1]

    clip = min([x1,y1,im_sz[0]-y2, im_sz[1]-x2])
    pad = 0
    if clip<=0:
        pad = abs(clip)+1
        map_get = np.zeros((int(im_sz[0]+2*pad), int(im_sz[1]+2*pad)))
        x1 = x1+pad
        x2 = x2+pad
        y1 = y1+pad
        y2 = y2+pad
    x2 = int(x1) + mask.shape[1]
    y2 = int(y1) + mask.shape[0]
    if x2 > im_sz[1]-1:
        x2 = im_sz[1]-1
        x1 = im_sz[1]-1-mask.shape[1]
    if y2 > im_sz[0]-1:
        y2 = im_sz[0]-1
        y1 = im_sz[0]-1-mask.shape[0]
    map_get[int(y1):int(y2),int(x1):int(x2)] = mask
    if clip<=0:
        map_get = map_get[int(pad+1):int(map_get.shape[0]-pad),int(pad+1):int(map_get.shape[1]-pad)]
    map_tmp = np.copy(map_get)
    map_get_out,_,_,_ = ext_roi(map_get, location, [0,0], roi_size, s)
    #map_get = map_get*255.0
    #map_get = map_get.astype('uint8')
    map_get = cv2.resize(map_get_out[:,:],(fea_sz[0],fea_sz[1]))
    map_get = map_get.astype('float32')
    if np.max(map_get) > 0:
        map_get = map_get/np.max(map_get)
    return map_get