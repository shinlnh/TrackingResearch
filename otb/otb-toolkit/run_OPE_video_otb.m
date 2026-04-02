function run_OPE_video_otb(mode, out_dir)
% Generate OTB OPE success plot for video validation.
% mode: 'only_mytracker' or 'full_tracker'

if nargin < 1 || isempty(mode)
    mode = 'full_tracker';
end

startup;

toolkit_path = get_global_variable('toolkit_path');
repo_root = fileparts(fileparts(toolkit_path));

if nargin < 2 || isempty(out_dir)
    out_dir = fullfile(repo_root, 'overall_result', 'video', 'otb', mode, 'success_plot');
end
ensure_dir(out_dir);
perfmat_path = fullfile(out_dir, 'tmp_perfmat');
ensure_dir(perfmat_path);

switch mode
    case 'only_mytracker'
        selected_names = {'MyTracker'};
    case 'full_tracker'
        baseline_names = { ...
            'MDNet', 'CCOT', 'DeepSRDCF', 'SRDCFdecon', 'SRDCF', 'Staple', ...
            'CSRT', 'HDT', 'CF2', 'LCT', 'CNN-SVM', 'SAMF', 'MEEM', 'DSST', 'KCF'};
        custom_names = {'MyTracker', 'ToMP-50'};
        selected_names = [baseline_names, custom_names];
    otherwise
        error('Unsupported mode: %s', mode);
end

all_trackers = config_trackers;
trackers = cell(1, numel(selected_names));
for i = 1:numel(selected_names)
    trackers{i} = get_or_create_tracker(all_trackers, selected_names{i});
end

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

OPE_perfmat_aligned(sequences, trackers, results_path, perfmat_path);
draw_success_plot_named(perfmat_path, trackers, linespecs, fullfile(out_dir, 'success_plot.png'));
write_success_auc_csv(perfmat_path, trackers, fullfile(out_dir, 'success_plot_scores.csv'));
fprintf('Saved OTB success plot to %s\n', out_dir);
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


function draw_success_plot_named(perfmat_path, trackers, linespecs, out_path)
threshold_set_overlap = 0:0.05:1;
perfmat_file = fullfile(perfmat_path, 'perfplot_curves_OPE.mat');
data = load(perfmat_file);
success_curve = data.success_curve;

ntrk = size(success_curve, 1);
nseq = size(success_curve, 2);
auc = cellfun(@mean, success_curve);
auc = mean(auc, 2);

legend_names = cell(ntrk, 1);
for i = 1:ntrk
    legend_names{i} = [trackers{i}.namePaper ' [' num2str(auc(i), '%.3f') ']'];
end

[~, rank] = sort(auc, 'descend');
success = reshape(cell2mat(success_curve), ntrk, length(threshold_set_overlap), nseq);
success = squeeze(mean(success, 3));

h = figure;
hold on;
for idx = 1:numel(rank)
    itrk = rank(idx);
    ls = linespecs{itrk};
    plot(threshold_set_overlap, success(itrk, :), 'Color', ls.color, ...
        'LineStyle', ls.lineStyle, 'LineWidth', ls.lineWidth);
end

legend(legend_names(rank), 'Location', 'southwest');
box on;
title('Success plots of OPE');
xlabel('Overlap threshold');
ylabel('Success rate');
set(gca, 'FontSize', 14);
saveas(h, out_path, 'png');
close(h);
end


function write_success_auc_csv(perfmat_path, trackers, csv_path)
perfmat_file = fullfile(perfmat_path, 'perfplot_curves_OPE.mat');
data = load(perfmat_file);
success_curve = data.success_curve;
auc = cellfun(@mean, success_curve);
auc = mean(auc, 2);
[~, rank] = sort(auc, 'descend');

fid = fopen(csv_path, 'w');
if fid < 0
    error('Failed to write CSV: %s', csv_path);
end
cleanup_obj = onCleanup(@() fclose(fid)); %#ok<NASGU>

fprintf(fid, 'tracker,success_rate,rank\n');
for idx = 1:numel(rank)
    itrk = rank(idx);
    fprintf(fid, '%s,%.6f,%d\n', trackers{itrk}.namePaper, auc(itrk), idx);
end
end


function ensure_dir(path_dir)
if exist(path_dir, 'dir') ~= 7
    mkdir(path_dir);
end
end
