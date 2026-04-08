% clc
% clear 
% close all

function rect_result = run_TLD_demo(video_path, video)

seq.name = video;

gt = dlmread([video_path '/groundtruth.txt']);

seq.init_rect = gt(1,:);

% resize_ratio = 2;

% % resize for hat-19
% if resize_ratio > 1
%     seq.init_rect = seq.init_rect * resize_ratio;
% end

imgs = dir([video_path '/img/*.jpg']);

for i = 1:numel(imgs)
    seq.s_frames{i} = [video_path '/img/' imgs(i).name];
end

% addpath('tld', 'utils_tld', 'img_tld', 'mex_tld', 'bbox_tld');

visualization = 0;
results = run_TLD_tracker(seq, visualization);
rect_result = results.res;
% if resize_ratio > 1
%     rect_result = rect_result / resize_ratio;
% end
end