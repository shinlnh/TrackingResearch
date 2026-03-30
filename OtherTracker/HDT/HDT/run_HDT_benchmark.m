function results = run_HDT_benchmark(seq, res_path, bSaveImage)
%RUN_HDT_BENCHMARK Headless OTB benchmark entry for HDT.

tracker_root = fileparts(mfilename('fullpath'));
pathModel = fullfile(tracker_root, 'imagenet-vgg-verydeep-19.mat');

if ~exist(pathModel, 'file')
    error('HDT:modelMissing', 'Missing VGG model file: %s', pathModel);
end

img_files = seq.s_frames;
rect_anno = load_rect_annotations(seq);

init_rect = seq.init_rect;
target_sz = [init_rect(4), init_rect(3)];
pos = [init_rect(2), init_rect(1)] + floor(target_sz/2);

padding = struct('generic', 2.2, 'large', 1, 'height', 0.4);
lambda = 1e-4;
output_sigma_factor = 0.1;
interp_factor = 0.01;
cell_size = 4;
show_visualization = 0;
bSaveImage = 0; %#ok<NASGU>

elapsed_time = tic;
fprintf('[HDT] %s start frames=%d\n', seq.name, numel(img_files));
positions = tracker_ensemble(img_files, pos, target_sz, ...
    padding, lambda, output_sigma_factor, interp_factor, ...
    cell_size, show_visualization, rect_anno, false, pathModel, seq.name);
elapsed_time = toc(elapsed_time);
fprintf('[HDT] %s done total_time=%.1fs fps=%.4f\n', seq.name, elapsed_time, numel(img_files) / max(elapsed_time, eps));

rects = zeros(size(positions, 1), 4);
rects(:,1) = positions(:,2) - target_sz(2)/2;
rects(:,2) = positions(:,1) - target_sz(1)/2;
rects(:,3) = target_sz(2);
rects(:,4) = target_sz(1);

results.type = 'rect';
results.res = rects;
results.fps = numel(img_files) / elapsed_time;
end

function rect_anno = load_rect_annotations(seq)
parts = strsplit(seq.name, '-');
anno_file = fullfile(seq.path, 'groundtruth_rect.txt');
if numel(parts) == 2
    anno_file = fullfile(seq.path, ['groundtruth_rect.' parts{2} '.txt']);
end

if ~exist(anno_file, 'file')
    fallback_file = fullfile(seq.path, 'groundtruth.txt');
    if exist(fallback_file, 'file')
        anno_file = fallback_file;
    end
end

rect_anno = readmatrix(anno_file, 'FileType', 'text');
if isempty(rect_anno) || size(rect_anno, 2) < 4 || any(isnan(rect_anno(:, 1:min(end, 4))), 'all')
    raw_text = fileread(anno_file);
    raw_text = strrep(raw_text, ',', ' ');
    raw_text = strrep(raw_text, sprintf('\t'), ' ');
    raw_text = regexprep(raw_text, '\r', ' ');
    values = sscanf(raw_text, '%f');
    rect_anno = reshape(values, 4, []).';
end
rect_anno = rect_anno(:, 1:4);

start_offset = 1;
if isfield(seq, 'startFrame') && isfield(seq, 'annoBegin')
    start_offset = seq.startFrame - seq.annoBegin + 1;
end

start_offset = max(1, start_offset);
last_index = min(size(rect_anno, 1), start_offset + numel(seq.s_frames) - 1);
rect_anno = rect_anno(start_offset:last_index, :);

if size(rect_anno, 1) < numel(seq.s_frames)
    pad = repmat(rect_anno(end, :), numel(seq.s_frames) - size(rect_anno, 1), 1);
    rect_anno = [rect_anno; pad];
end
end
