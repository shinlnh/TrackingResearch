function tracked_bb = run_cfnet_lasot(seq_dir, ~)
%RUN_CFNET_LASOT Run CFNet on a single LaSOT sequence.

repo_root = fileparts(fileparts(fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))))));
cfnet_root = fullfile(repo_root, 'OtherTracker', 'verified', 'cfnet-master');

img_listing = dir(fullfile(seq_dir, 'img', '*.jpg'));
if isempty(img_listing)
    img_listing = dir(fullfile(seq_dir, 'img', '*.png'));
end
assert(~isempty(img_listing), 'No image files found for %s', seq_dir);

gt = dlmread(fullfile(seq_dir, 'groundtruth.txt'));
[cx, cy, w, h] = get_axis_aligned_BB(gt(1, :));

frame_paths = cellfun(@(name) fullfile(seq_dir, 'img', name), {img_listing.name}, 'UniformOutput', false);

paths = struct();
paths.net_base = [fullfile(cfnet_root, 'pretrained', 'networks') filesep];
paths.eval_set_base = [fullfile(repo_root, 'ls', 'lasot') filesep];
paths.stats = fullfile(cfnet_root, 'data', 'ILSVRC2015.stats.mat');

tracker_params = struct();
tracker_params.visualization = false;
tracker_params.gpus = 1;
tracker_params.join = struct('method', 'corrfilt');
tracker_params.net = 'cfnet-conv2_e80.mat';
tracker_params.net_gray = 'cfnet-conv2_gray_e40.mat';
tracker_params.scaleStep = 1.0300;
tracker_params.scalePenalty = 0.9800;
tracker_params.scaleLR = 0.5900;
tracker_params.wInfluence = 0.2500;
tracker_params.zLR = 0.0050;
tracker_params.paths = paths;
tracker_params.imgFiles = vl_imreadjpeg(frame_paths, 'numThreads', 8);
tracker_params.targetPosition = [cy cx];
tracker_params.targetSize = round([h w]);

try
    parallel.gpu.enableCUDAForwardCompatibility(true);
catch
end

[tracked_bb, ~] = tracker(tracker_params);
tracked_bb = double(tracked_bb);
end
