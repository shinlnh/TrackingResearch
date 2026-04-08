
% run_tracker.m

% close all;
% clear all;

%choose the path to the videos (you'll be able to choose one with the GUI)
% base_path = 'sequences/';

function rect_results = run_tracker(tmp_video_path, tmp_video)

%parameters according to the paper
params.padding = 1.0;         			   % extra area surrounding the target
params.output_sigma_factor = 1/16;		   % spatial bandwidth (proportional to target)
params.sigma = 0.2;         			   % gaussian kernel bandwidth
params.lambda = 1e-2;					   % regularization (denoted "lambda" in the paper)
params.learning_rate = 0.075;			   % learning rate for appearance model update scheme (denoted "gamma" in the paper)
params.compression_learning_rate = 0.15;   % learning rate for the adaptive dimensionality reduction (denoted "mu" in the paper)
params.non_compressed_features = {'gray'}; % features that are not compressed, a cell with strings (possible choices: 'gray', 'cn')
params.compressed_features = {'cn'};       % features that are compressed, a cell with strings (possible choices: 'gray', 'cn')
params.num_compressed_dim = 2;             % the dimensionality of the compressed features

params.visualization = 0;

[img_files, pos, target_sz, ground_truth, video_path] = load_video_info(tmp_video_path);

params.init_pos = floor(pos);
params.wsize = floor(target_sz);
params.img_files = img_files;
params.video_path = video_path;

[positions, rect_results, fps] = color_tracker(params);
end