function tracked_bb = run_PTAV_lasot(seq_dir, video)
%RUN_PTAV_LASOT Run serial PTAV on a single LaSOT sequence using a Python OpenCV DNN backend.

if nargin < 2 || isempty(video)
    [~, video] = fileparts(seq_dir);
end

addpath('util');
ptav_init_backend();

gt = dlmread(fullfile(seq_dir, 'groundtruth.txt'));
init_box = gt(1, :);
target_sz = [gt(1, 4), gt(1, 3)];
pos = [gt(1, 2), gt(1, 1)] + floor(target_sz / 2);

img_dir = fullfile(seq_dir, 'img');
listing = dir(fullfile(img_dir, '*.jpg'));
if isempty(listing)
    listing = dir(fullfile(img_dir, '*.png'));
end
assert(~isempty(listing), 'PTAV:missingFrames', 'No image frames found in %s', img_dir);
img_files = sort({listing.name});
video_path = [img_dir filesep];

verify_param = init_verifier(img_files, video, video_path, init_box);
input_roi = get_rois(verify_param.init_box, verify_param.imageSz, verify_param.firstframe);
verify_param.firstframe_feat = ptav_extract_features( ...
    verify_param.firstframe, input_roi, verify_param.pixel_means, verify_param.imageSz);

params = struct();
fDSST_param;
params.visualization = 0;
params.init_pos = floor(pos);
params.wsize = floor(target_sz);
params.s_frames = img_files;
params.video_path = video_path;

results = PTAV(params, verify_param);
tracked_bb = double(results.res);
end
