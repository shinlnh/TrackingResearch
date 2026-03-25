startup;

trackers = { ...
    struct('name', 'MyTrackerECO', 'namePaper', 'MyTracker'), ...
    struct('name', 'Staple', 'namePaper', 'Staple'), ...
    struct('name', 'SiamFC', 'namePaper', 'SiamFC'), ...
    struct('name', 'CF2', 'namePaper', 'CF2'), ...
    struct('name', 'LCT', 'namePaper', 'LCT'), ...
    struct('name', 'SAMF', 'namePaper', 'SAMF'), ...
    struct('name', 'MEEM', 'namePaper', 'MEEM'), ...
    struct('name', 'DSST', 'namePaper', 'DSST'), ...
    struct('name', 'KCF', 'namePaper', 'KCF') ...
};

linespecs = { ...
    get_linespec('--', 'r'), ...
    get_linespec('-', 'r'), ...
    get_linespec('--', 'g'), ...
    get_linespec('-', 'k'), ...
    get_linespec('-', 'y'), ...
    get_linespec('-', 'm'), ...
    get_linespec('-', 'c'), ...
    get_linespec('-', 'b'), ...
    get_linespec('-', 'g') ...
};

toolkit_path = get_global_variable('toolkit_path');
results_path = fullfile(toolkit_path, 'results', 'OPE');
perfmat_path = fullfile(toolkit_path, 'perfmat', 'OPE');
figure_path = fullfile(toolkit_path, 'figs', 'OPE');

dataset_root = fullfile(toolkit_path, '..', 'otb100');
if exist(dataset_root, 'dir') ~= 7
    dataset_root = fullfile(toolkit_path, 'sequences');
end

sequences = build_sequences_with_complete_results(toolkit_path, dataset_root, trackers);
fprintf('Common evaluated sequences: %d\n', numel(sequences));

if isempty(sequences)
    error('No sequence has complete results and valid annotations.');
end

OPE_perfmat_aligned(sequences, trackers, results_path, perfmat_path);

out_path = fullfile(figure_path, 'success_plot_mytracker_eco_compare.png');
draw_success_plot_named(perfmat_path, trackers, linespecs, out_path);
fprintf('Saved ECO comparison figure: %s\n', out_path);


function linespec = get_linespec(style, color)
linespec = struct('lineWidth', 3, 'lineStyle', style, 'color', color);
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

nameTrkAll = cell(1, ntrk);
for i = 1:ntrk
    nameTrkAll{i} = trackers{i}.name;
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
