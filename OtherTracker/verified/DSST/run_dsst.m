function results = run_dsst(seq, res_path, bSaveImage)
%RUN_DSST OTB harness wrapper for the original DSST MATLAB code.

if nargin < 1 || isempty(seq)
    error('DSST:runWrapper', 'Sequence input is required.');
end

if nargin < 2
    res_path = ''; %#ok<NASGU>
end

if nargin < 3 || isempty(bSaveImage)
    bSaveImage = false; %#ok<NASGU>
end

global dsst_sequence_name;
dsst_sequence_name = seq.name;

params = struct();
params.padding = 1.0;
params.output_sigma_factor = 1/16;
params.scale_sigma_factor = 1/4;
params.lambda = 1e-2;
params.learning_rate = 0.025;
params.number_of_scales = 33;
params.scale_step = 1.02;
params.scale_model_max_area = 512;
params.visualization = 0;

params.init_pos = floor(seq.init_rect(1, [2, 1])) + floor(seq.init_rect(1, [4, 3]) / 2);
params.wsize = floor(seq.init_rect(1, [4, 3]));

img_files = cell(numel(seq.s_frames), 1);
for i = 1:numel(seq.s_frames)
    [~, name, ext] = fileparts(seq.s_frames{i});
    img_files{i} = [name ext];
end

params.img_files = img_files;
params.video_path = [fileparts(seq.s_frames{1}) filesep];

[positions, fps] = dsst(params);

rects = zeros(size(positions, 1), 4);
rects(:, 1) = positions(:, 2) - positions(:, 4) / 2;
rects(:, 2) = positions(:, 1) - positions(:, 3) / 2;
rects(:, 3) = positions(:, 4);
rects(:, 4) = positions(:, 3);

results = struct();
results.type = 'rect';
results.res = rects;
results.fps = fps;
end
