% Parallel Tracking and Verifying: A Framework for Real-Time and High Accuracy Visual Tracking, ICCV, 2017
% Copyright @ Heng Fan and Haibin Ling, 2017
%
% If you have any questions, please feel free to contact Heng Fan (hengfan@temple.edu)

clc;
clear;
close all;

addpath('util\');
addpath('C:\Users\HengLan\Desktop\caffe-windows-master\matlab');   % path to your matcaffe

base_path = 'E:\TrackingBenchmark15\';

% deploy document and caffe model for the verifier
def       = 'siamese_networks\deploy.prototxt';        
weight    = 'siamese_networks\similarity.caffemodel';  

% load verifying networks
global net;
caffe.set_mode_gpu();
caffe.set_device(1);
net = caffe.Net(def, weight, 'test');

% load parameters for fDSST tracker
fDSST_param;

% load videos
videos = dir(base_path);
videos([1, 2]) = [];

for i = 1:1
    video = videos(i).name;
    fprintf('processing video %s ......\n', video);
    
    % load images
    video_path = [base_path video '\'];
    [img_files, pos, target_sz, ground_truth, video_path, init_box] = load_video_info(video_path);

    % parameters for verifier
    verify_param = init_verifier(img_files, video, video_path, init_box);
    
    % prepare image for caffe networks
    input_im     = prepare_image(verify_param.firstframe, verify_param.imageSz, verify_param.pixel_means);
    input_roi    = get_rois(verify_param.init_box, verify_param.imageSz, verify_param.firstframe);

    % extract feature for object in the first frame
    firstframe_input_blobs    = cell(2, 1);
    firstframe_input_blobs{1} = input_im;
    net.blobs('rois').reshape([5, size(input_roi, 1)]);
    firstframe_input_blobs{2} = input_roi';

    blobs_out                    = net.forward(firstframe_input_blobs);
    verify_param.firstframe_feat = squeeze(blobs_out{1});      % feature for the target

    params.init_pos = floor(pos);
    params.wsize = floor(target_sz);
    params.s_frames = img_files;
    params.video_path = video_path;

    % do tracking
    results = PTAV(params, verify_param);

    positions = results.res;
    tmp_pos    = positions(:, [2, 1]) + positions(:, [4, 3])/2;
    precisions = precision_plot(tmp_pos, ground_truth);
    fprintf('%12s - Precision (20px):% 1.3f\n', video, precisions(20));
end

% delete caffe model from memory
caffe.reset_all();

% you may need to manually remove the log folder and WARNING file