import scipy.io as sio
import numpy as np
from func import create_labels
import random
import cv2

id = sio.loadmat('imdb_id.mat')
id = id['id']
n_valid_objects = sio.loadmat('imdb_n_valid_objects.mat')
n_valid_objects = n_valid_objects['n_valid_objects']
nframes = sio.loadmat('imdb_nframes.mat')
nframes = nframes['nframes']
objects1 = sio.loadmat('imdb_objects_1.mat')
objects1 = objects1['objects_1']
objects2 = sio.loadmat('imdb_objects_2.mat')
objects2 = objects2['objects_1']
objects = np.concatenate((objects1,objects2),axis=1)
del objects1
del objects2
path = sio.loadmat('imdb_path.mat')
path = path['path']
valid_per_trackid = sio.loadmat('imdb_valid_per_trackid.mat')
valid_per_trackid = valid_per_trackid['valid_per_trackid']
valid_trackids = sio.loadmat('imdb_valid_trackids.mat')
valid_trackids = valid_trackids['valid_trackids']

imdb_video = {
    'id':id,
    'n_valid_objects':n_valid_objects,
    'nframes':nframes,
    'objects':objects,
    'path':path,
    'valid_per_trackid':valid_per_trackid,
    'valid_trackids': valid_trackids
}

del id
del n_valid_objects
del nframes
del objects
del path
del valid_per_trackid
del valid_trackids

size_dataset = imdb_video['id'].shape[1]
size_validation = np.round(0.1*size_dataset)
size_training = size_dataset - size_validation
imdb_video_set = np.zeros((size_dataset,))
imdb_video_set[:size_training] = 1
imdb_video_set[size_training:] = 2
imdb_video['set'] = imdb_video_set

imdb_id = np.arange(0,53200)
n_pairs_train = int(np.round(53200 * (1-0.1)))
imdb_images_set = np.zeros((53200,))
imdb_images_set[:n_pairs_train] = 1
imdb_images_set[n_pairs_train:] = 2
imdb = {
    'id': imdb_id,
    'images': imdb_images_set
}

resp_sz = [15,15]
resp_stride=8

examplerSize = 127
instanceSize = 239

pos_eltwise, instanceWeight = create_labels(resp_sz,16 / resp_stride, 0 / resp_stride)

index_tmp = range(0,n_pairs_train)
random.shuffle(index_tmp)
index_tmp = np.asarray(index_tmp)
state = {'train': index_tmp.copy()}
index_tmp = range(0,53200 - n_pairs_train)
random.shuffle(index_tmp)
index_tmp = np.asarray(index_tmp)
state['val'] = index_tmp.copy()

batch_size = 8
batch = state['train'][0:batch_size] #determine iteration
batch_set = imdb['images'][batch[0]]

ids_set = [i for i,val in enumerate(imdb_video['set'] == batch_set) if val == True]
rnd_videos = random.sample(ids_set,batch_size)
data_dir = '/home/xiaobai/Documents/ILSVRC2015_crops/Data/VID/train/'
sizes_z = np.zeros((2,batch_size))
sizes_x = np.zeros((2,batch_size))
for b in range(batch_size):
    rand_vid = rnd_videos[b]
    #choose_pos_pair
    valid_trackids = imdb_video['valid_trackids'][:,rand_vid]
    valid_trackids = [i for i,val in enumerate(valid_trackids) if val > 1]
    rand_trackid_z = random.sample(valid_trackids,1)[0]
    rand_z = imdb_video['valid_per_trackid'][rand_trackid_z,rand_vid][0,:]
    rand_z = random.sample(rand_z,1)[0]
    possible_x_pos = range(0,imdb_video['valid_per_trackid'][rand_trackid_z,rand_vid].shape[1])
    rand_z_pos = [i for i,val in enumerate(imdb_video['valid_per_trackid'][rand_trackid_z,rand_vid][0,:]) if val == rand_z][0]
    possible_x_pos = possible_x_pos[np.max([0,rand_z_pos - 100]):(rand_z_pos-1)]+possible_x_pos[(rand_z_pos+1):np.min([rand_z_pos+100,len(possible_x_pos)])]
    possible_x = imdb_video['valid_per_trackid'][rand_trackid_z,rand_vid][0,possible_x_pos]
    rand_x = random.sample(possible_x,1)[0]
    z = imdb_video['objects'][0,rand_vid][0,rand_z]
    x = imdb_video['objects'][0,rand_vid][0,rand_x]

    crops_z_string = data_dir + ''.join(z['frame_path'][0][0])[:(len(''.join(z['frame_path'][0][0]))-5)] + '.' + '%02d'%z['track_id'][0][0][0,0] + '.crop.z.jpg'
    crops_x_string = data_dir + ''.join(x['frame_path'][0][0])[:(len(''.join(x['frame_path'][0][0]))-5)] + '.' + '%02d'%x['track_id'][0][0][0,0] + '.crop.x.jpg'

    crops_z = cv2.imread(crops_z_string)
    crops_x = cv2.imread(crops_x_string)

    if b == 0:
        imout_z = crops_z
        imout_z = np.expand_dims(imout_z,0)
        imout_x = crops_x
        imout_x = np.expand_dims(imout_x,0)
    else:
        imout_z = np.concatenate((imout_z,np.expand_dims(crops_z,0)),axis=0)
        imout_x = np.concatenate((imout_x,np.expand_dims(crops_x,0)),axis=0)

print 'hehe'

