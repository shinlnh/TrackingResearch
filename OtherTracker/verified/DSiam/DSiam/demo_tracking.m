%% DEMO_TRACKING
function rect_results = demo_tracking(video_path, video)

rootdir = '/home/lab/Desktop/test/DSiam/';

% seq.len = seq.endFrame - seq.startFrame + 1;
% seq.s_frames = cell(seq.len,1);
% nz	= strcat('%0',num2str(seq.nz),'d'); %number of zeros in the name of image
% for i=1:seq.len
%     image_no = seq.startFrame + (i-1);
%     id = sprintf(nz,image_no);
%     seq.s_frames{i} = strcat(seq.path,'img/',id,'.',seq.ext);
% end
% 
% rect_anno = dlmread([seq.path 'groundtruth_rect.txt']);
% seq.init_rect = rect_anno(seq.startFrame,:);

seq = struct('name','skiing','path',video_path,'startFrame',1, ...
            'endFrame',0,'nz',8,'ext','jpg');

imgs = dir([video_path '/img/*.jpg']);
for i = 1:numel(imgs)
    seq.s_frames{i} = [video_path '/img/' imgs(i).name];
end
seq.len = numel(imgs);

rect_anno = dlmread([video_path '/groundtruth.txt']);
seq.init_rect = rect_anno(1,:);

isDisplay = 0;

% the pretrained network for Dynamic Siamese Network 
netname = 'siamfc';
% '1res' denotes the multi-layer DSiam (DSiamM in paper) and uses two layers for tracking
% '0res' denotes the single-layer DSiam (DSiam in paper) and uses the last layer for tracking
nettype = '1res';
results = run_DSiam_tracker(seq,[],isDisplay,rootdir,netname,nettype);
rect_results = results.res;
end