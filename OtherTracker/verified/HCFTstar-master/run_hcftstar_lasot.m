function tracked_bb = run_hcftstar_lasot(base_path, video)
%RUN_HCFTSTAR_LASOT Run HCFT on a single LaSOT sequence.

show_visualization = false;

padding = struct('generic', 1.8, 'large', 1, 'height', 0.4);
lambda = 1e-4;
output_sigma_factor = 0.1;
interp_factor = 0.01;
cell_size = 4;

config = struct();
config.kernel_sigma = 1;
config.motion_thresh = 0.181;
config.appearance_thresh = 0.38;
config.features = struct();
config.features.hog_orientations = 9;
config.features.cell_size = 4;
config.features.window_size = 6;
config.features.nbins = 8;

global enableGPU;
enableGPU = true;
try
    parallel.gpu.enableCUDAForwardCompatibility(true);
catch
end

[img_files, pos, target_sz, video_path] = load_lasot_video_info(base_path, video);
[~, ~, rect_position] = tracker_HCFTstar(video_path, img_files, pos, target_sz, ...
    padding, lambda, output_sigma_factor, interp_factor, ...
    cell_size, show_visualization, config);

tracked_bb = double(rect_position);
end

function [img_files, pos, target_sz, video_path] = load_lasot_video_info(base_path, video)
seq_path = fullfile(base_path, video);
gt_path = fullfile(seq_path, 'groundtruth.txt');
assert(exist(gt_path, 'file') == 2, 'Missing LaSOT groundtruth: %s', gt_path);

ground_truth = dlmread(gt_path);
target_sz = [ground_truth(1, 4), ground_truth(1, 3)];
pos = [ground_truth(1, 2), ground_truth(1, 1)] + floor(target_sz / 2);

video_path = fullfile(seq_path, 'img');
if video_path(end) ~= '/' && video_path(end) ~= '\'
    video_path(end + 1) = '/';
end

img_listing = dir(fullfile(video_path, '*.jpg'));
if isempty(img_listing)
    img_listing = dir(fullfile(video_path, '*.png'));
end
assert(~isempty(img_listing), 'No image files found in %s', video_path);
img_files = sort_nat({img_listing.name});
end

function out = sort_nat(in)
[~, order] = sort(lower(in));
out = in(order);
end
