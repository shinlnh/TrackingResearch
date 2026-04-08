
% This demo script runs the STRCF tracker with deep features on the
% included "Human3" video.
% clc
% clear
% close all

% Add paths


%  Load video information
% base_path  =  './sequences';
% video = 'Human3';

function rect_result = demo_STRCF(video_path, video)

% video_path = [base_path '/' video];
[seq, ~] = load_video_info(video_path);

% Run STRCF
results = run_STRCF_code(seq);
rect_result = results.res;
end