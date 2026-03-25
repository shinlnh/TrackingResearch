startup;

% Realtime-oriented trackers available in this toolkit.
realtime_names = {'Staple', 'KCF', 'DSST', 'LCT', 'SAMF', 'MEEM', 'CF2'};

% Add your tracker here.
custom_names = {'MyTracker', 'SiamFC'};

selected_names = [realtime_names, custom_names];

all_trackers = config_trackers;
trackers = cell(1, numel(selected_names));
for i = 1:numel(selected_names)
    trackers{i} = get_or_create_tracker(all_trackers, selected_names{i});
end

toolkit_path = get_global_variable('toolkit_path');
results_path = fullfile(toolkit_path, 'results', 'OPE');

% Use local OTB root if toolkit/sequences is incomplete.
dataset_root = 'F:\AIDataset\CV\Tracking\otb100';
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

% External pure CSRT reference (paper-style rerun) from CSRTResearch.
ref_csv = 'E:\Programming\C\C2P\Project\CSRTResearch\pure_csrt\auc_pure_rerun.csv';
ref_curve_csv = 'E:\Programming\C\C2P\Project\CSRTResearch\pure_csrt\success_curve_rerun.csv';
ref_label = 'PureCSRT (paper)';
ref_auc = read_overall_auc_from_csv(ref_csv);
ref_curve = read_success_curve_csv(ref_curve_csv);

% Evaluate and draw OPE plots from existing result files.
OPE_perfmat_aligned(sequences, trackers, results_path, fullfile(toolkit_path, 'perfmat', 'OPE'));
OPE_drawplot(sequences, trackers, linespecs);

fig_dir = fullfile(toolkit_path, 'figs', 'OPE');
src_fig = fullfile(fig_dir, 'success_plot.png');
dst_fig = fullfile(fig_dir, 'success_plot_realtime_compare.png');
if exist(src_fig, 'file') == 2
    copyfile(src_fig, dst_fig);
    fprintf('Saved realtime comparison figure: %s\n', dst_fig);
end

% Draw an augmented figure that includes external pure CSRT reference.
perfmat_file = fullfile(toolkit_path, 'perfmat', 'OPE', 'perfplot_curves_OPE.mat');
dst_aug_fig = fullfile(fig_dir, 'success_plot_realtime_compare_with_pure_csrt.png');
draw_augmented_realtime_plot(perfmat_file, linespecs, ref_auc, ref_curve, ref_label, dst_aug_fig);
fprintf('Saved augmented realtime figure: %s\n', dst_aug_fig);


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
    sequences{end + 1} = seq; %#ok<AGROW>
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
    nameTrkAll{end + 1} = trackers{i}.name; %#ok<AGROW>
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


function ref_auc = read_overall_auc_from_csv(csv_path)
ref_auc = NaN;
if exist(csv_path, 'file') ~= 2
    fprintf('Warning: CSV not found: %s\n', csv_path);
    return;
end

fid = fopen(csv_path, 'r');
if fid < 0
    fprintf('Warning: Cannot open CSV: %s\n', csv_path);
    return;
end
cleanup_obj = onCleanup(@() fclose(fid)); %#ok<NASGU>

header = fgetl(fid); %#ok<NASGU>
while true
    line = fgetl(fid);
    if ~ischar(line)
        break;
    end
    if isempty(strtrim(line))
        continue;
    end
    tokens = strsplit(line, ',');
    if numel(tokens) < 3
        continue;
    end
    if strcmp(strtrim(tokens{1}), 'OVERALL')
        val = str2double(tokens{3}); % auc_update
        if ~isnan(val)
            ref_auc = val;
        end
        return;
    end
end
end


function ref_curve = read_success_curve_csv(curve_csv_path)
ref_curve = struct('threshold', [], 'success', []);
if exist(curve_csv_path, 'file') ~= 2
    fprintf('Warning: success curve CSV not found: %s\n', curve_csv_path);
    return;
end

fid = fopen(curve_csv_path, 'r');
if fid < 0
    fprintf('Warning: Cannot open success curve CSV: %s\n', curve_csv_path);
    return;
end
cleanup_obj = onCleanup(@() fclose(fid)); %#ok<NASGU>

header = fgetl(fid); %#ok<NASGU>
ts = [];
ss = [];
while true
    line = fgetl(fid);
    if ~ischar(line)
        break;
    end
    if isempty(strtrim(line))
        continue;
    end
    tokens = strsplit(line, ',');
    if numel(tokens) < 2
        continue;
    end
    t = str2double(tokens{1});
    s = str2double(tokens{2});
    if ~isnan(t) && ~isnan(s)
        ts(end + 1) = t; %#ok<AGROW>
        ss(end + 1) = s; %#ok<AGROW>
    end
end

if ~isempty(ts)
    ref_curve.threshold = ts;
    ref_curve.success = ss;
end
end


function draw_augmented_realtime_plot(perfmat_file, linespecs, ref_auc, ref_curve, ref_label, out_path)
threshold_set_overlap = 0:0.05:1;

data = load(perfmat_file);
success_curve = data.success_curve;
nameTrkAll = data.nameTrkAll;

ntrk = size(success_curve, 1);
nseq = size(success_curve, 2);

auc = cellfun(@mean, success_curve);
auc = mean(auc, 2);

legend_names = cell(ntrk, 1);
for i = 1:ntrk
    legend_names{i} = [nameTrkAll{i} ' [' num2str(auc(i), '%.3f') ']'];
end
[~, rank] = sort(auc, 'descend');

success = reshape(cell2mat(success_curve), ntrk, length(threshold_set_overlap), nseq);
success = squeeze(mean(success, 3));

h = figure;
hold on;
for k = 1:numel(rank)
    itrk = rank(k);
    ls = linespecs{itrk};
    plot(threshold_set_overlap, success(itrk, :), 'Color', ls.color, ...
        'LineStyle', ls.lineStyle, 'LineWidth', ls.lineWidth);
end

legend_entries = legend_names(rank);

if ~isempty(ref_curve.threshold)
    plot(ref_curve.threshold, ref_curve.success, '--', 'Color', [0.35 0.35 0.35], 'LineWidth', 2.5);
    if ~isnan(ref_auc)
        legend_entries{end + 1} = [ref_label ' [' num2str(ref_auc, '%.3f') ']']; %#ok<AGROW>
    else
        legend_entries{end + 1} = ref_label; %#ok<AGROW>
    end
elseif ~isnan(ref_auc)
    plot(threshold_set_overlap, ref_auc * ones(size(threshold_set_overlap)), '--', ...
        'Color', [0.35 0.35 0.35], 'LineWidth', 2.5);
    legend_entries{end + 1} = [ref_label ' [' num2str(ref_auc, '%.3f') ']']; %#ok<AGROW>
end

legend(legend_entries, 'Location', 'southwest');
box on;
title('Success plots of OPE (Realtime + PureCSRT reference)');
xlabel('Overlap threshold');
ylabel('Success rate');
saveas(h, out_path, 'png');
close(h);
end
