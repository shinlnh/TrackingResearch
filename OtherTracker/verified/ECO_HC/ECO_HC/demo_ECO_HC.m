
% This demo script runs the ECO tracker with hand-crafted features on the
% included "Crossing" video.

% Add paths
% close all;
% clc;
% setup_paths();
function rect_result = demo_ECO_HC(video_path, video)
% Load video information
% video_path = 'sequences/face';
[seq, ~] = load_video_info(video_path);

% Run ECO
results = testing_ECO_HC(seq);
rect_result = results.res;
end