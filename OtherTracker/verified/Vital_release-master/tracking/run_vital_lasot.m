function tracked_bb = run_vital_lasot(video_path, ~)
%RUN_VITAL_LASOT Run VITAL on a single LaSOT sequence.

img_dir = fullfile(video_path, 'img');
image_info = dir(fullfile(img_dir, '*.jpg'));
if isempty(image_info)
    image_info = dir(fullfile(img_dir, '*.png'));
end
assert(~isempty(image_info), 'run_vital_lasot:noFrames', 'No image frames found in %s', img_dir);

[~, order] = sort({image_info.name});
image_info = image_info(order);
images = cell(numel(image_info), 1);
for i = 1:numel(image_info)
    images{i} = fullfile(img_dir, image_info(i).name);
end

ground_truth = dlmread(fullfile(video_path, 'groundtruth.txt'));
net = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'models', 'otbModel.mat');

global gpu;
gpu = true;
try
    parallel.gpu.enableCUDAForwardCompatibility(true);
catch
end

tracked_bb = double(vital_run(images, ground_truth(1, :), net, false));
end
