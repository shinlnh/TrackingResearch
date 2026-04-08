function [seq, ground_truth] = load_video_info(video_path)

ground_truth = dlmread([video_path '/groundtruth.txt']);

seq.init_rect = ground_truth(1,:);

img_path = [video_path '/img/'];

img_files = dir(fullfile(img_path, '*.jpg'));
seq.len = numel(img_files);
img_files = {img_files.name};
seq.s_frames = cellstr(img_files);

end

