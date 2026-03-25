startup;

% Baseline trackers shown in the common 16-tracker OTB OPE plot.
baseline_names = { ...
    'MDNet', 'CCOT', 'DeepSRDCF', 'SRDCFdecon', 'SRDCF', 'Staple', ...
    'HDT', 'CF2', 'LCT', 'CNN-SVM', 'SAMF', 'MEEM', 'DSST', 'KCF'};

% Replace the old custom tracker with the ECO-based MyTracker.
custom_names = {'MyTracker', 'ToMP_plus3'};

selected_names = [baseline_names, custom_names];

all_trackers = config_trackers;
trackers = cell(1, numel(selected_names));
for i = 1:numel(selected_names)
    trackers{i} = get_or_create_tracker(all_trackers, selected_names{i});
end

toolkit_path = get_global_variable('toolkit_path');
results_path = fullfile(toolkit_path, 'results', 'OPE');

dataset_root = fullfile(toolkit_path, '..', 'otb100');
if exist(dataset_root, 'dir') ~= 7
    dataset_root = fullfile(toolkit_path, 'sequences');
end

sequences = build_sequences_with_complete_results(toolkit_path, dataset_root, trackers);
fprintf('Common evaluated sequences: %d\n', numel(sequences));

if isempty(sequences)
    error('No sequence has complete results and valid annotations.');
end

linespecs = config_linespecs;
if numel(linespecs) < numel(trackers)
    error('Not enough line styles for the selected trackers.');
end
linespecs = linespecs(1:numel(trackers));

% Evaluate and draw OPE plots from existing result files using original toolkit plotting.
OPE_perfmat_aligned(sequences, trackers, results_path, fullfile(toolkit_path, 'perfmat', 'OPE'));
OPE_drawplot(sequences, trackers, linespecs);

fig_dir = fullfile(toolkit_path, 'figs', 'OPE');
src_fig = fullfile(fig_dir, 'success_plot.png');
dst_fig = fullfile(fig_dir, 'success_plot_mytracker_eco_full16.png');
if exist(src_fig, 'file') == 2
    copyfile(src_fig, dst_fig);
    fprintf('Saved full-16 ECO figure: %s\n', dst_fig);
end


function tracker = get_or_create_tracker(trackers, tracker_name)
for i = 1:numel(trackers)
    if strcmp(trackers{i}.name, tracker_name)
        tracker = trackers{i};
        return;
    end
end

tracker = struct('name', tracker_name, 'namePaper', tracker_name);
end


function sequences = build_sequences_with_complete_results(toolkit_path, dataset_root, trackers)
results_path = fullfile(toolkit_path, 'results', 'OPE');
seq_file = fullfile(toolkit_path, 'sequences', 'SEQUENCES');
fid = fopen(seq_file, 'r');
if fid < 0
    error('Cannot open SEQUENCES file: %s', seq_file);
end

cleanup_obj = onCleanup(@() fclose(fid)); %#ok<NASGU>
sequences = {};

while true
    line = fgetl(fid);
    if ~ischar(line)
        break;
    end
    seq_name = strtrim(line);
    if isempty(seq_name)
        continue;
    end

    has_all_results = true;
    for itrk = 1:numel(trackers)
        result_file = fullfile(results_path, [seq_name '_' trackers{itrk}.name '.mat']);
        if exist(result_file, 'file') ~= 2
            has_all_results = false;
            break;
        end
    end
    if ~has_all_results
        continue;
    end

    anno = load_otb_annotation(dataset_root, seq_name);
    if isempty(anno)
        continue;
    end

    seq = struct('name', seq_name, 'annos', anno);
    sequences{end+1} = seq; %#ok<AGROW>
end
end


function anno = load_otb_annotation(dataset_root, seq_name)
anno = [];

parts = strsplit(seq_name, '-');
if numel(parts) == 2
    base_name = parts{1};
    split_id = parts{2};
    anno_file = fullfile(dataset_root, base_name, ['groundtruth_rect.' split_id '.txt']);
else
    base_name = seq_name;
    anno_file = fullfile(dataset_root, base_name, 'groundtruth_rect.txt');
end

if exist(anno_file, 'file') ~= 2
    return;
end

anno = dlmread(anno_file);
end


function OPE_perfmat_aligned(sequences, trackers, results_path, perfmat_path)
eval_type = 'OPE';
nseq = length(sequences);
ntrk = length(trackers);

nameTrkAll = {};
for i = 1:ntrk
    nameTrkAll{end+1} = trackers{i}.name; %#ok<AGROW>
end

success_curve = cell(ntrk, nseq);
precision_curve = cell(ntrk, nseq);

for iseq = 1:nseq
    for itrk = 1:ntrk
        fprintf('%5d_%-12s,%3d_%-20s\n', iseq, sequences{iseq}.name, itrk, trackers{itrk}.name);
        [success, precision] = perf_aligned(sequences{iseq}, trackers{itrk}, results_path);
        success_curve{itrk, iseq} = success;
        precision_curve{itrk, iseq} = precision;
    end
end

perfmat_file = fullfile(perfmat_path, ['perfplot_curves_' eval_type '.mat']);
save(perfmat_file, 'success_curve', 'precision_curve', 'nameTrkAll');
end


function [success, precision] = perf_aligned(sequence, tracker, results_path)
threshold_set_overlap = 0:0.05:1;
threshold_set_error = 0:50;

results_file = fullfile(results_path, [sequence.name '_' tracker.name '.mat']);
data = load(results_file);
res = data.results{1};
anno = align_annotation_for_result(sequence.annos, res);

[~, ~, err_coverage, err_center] = calcSeqErrRobust(res, anno);

success_num_overlap = zeros(1, length(threshold_set_overlap));
for i = 1:length(threshold_set_overlap)
    success_num_overlap(i) = sum(err_coverage > threshold_set_overlap(i));
end

success_num_err = zeros(1, length(threshold_set_error));
for i = 1:length(threshold_set_error)
    success_num_err(i) = sum(err_center <= threshold_set_error(i));
end

len = size(anno, 1);
success = success_num_overlap / (len + eps);
precision = success_num_err / (len + eps);
end


function anno = align_annotation_for_result(anno_in, res)
anno = anno_in;
anno_len = size(anno, 1);
target_len = res.len;
start_frame = 1;
if isfield(res, 'startFrame')
    start_frame = res.startFrame;
end

if anno_len == target_len
    return;
end

if start_frame > 1 && (anno_len - start_frame + 1) >= target_len
    end_frame = start_frame + target_len - 1;
    anno = anno(start_frame:end_frame, :);
    return;
end

if anno_len > target_len
    anno = anno(1:target_len, :);
    return;
end

if anno_len < target_len
    pad = repmat(anno(end, :), target_len - anno_len, 1);
    anno = [anno; pad];
end
end
