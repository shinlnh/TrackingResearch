function plot_sota_comparison()
% PLOT_SOTA_COMPARISON
% Generates comparison charts for MyECOTracker vs OSTrack vs STARK on
% LaSOT head-tail-40 and OTB-100. Reads summary.csv + per_sequence_metrics.csv
% produced by the Python runners.
%
% Output: PNG + .fig under overall_result/sota_comparison/matlab/

repo_root = fileparts(fileparts(mfilename('fullpath')));
out_dir = fullfile(repo_root, 'overall_result', 'sota_comparison', 'matlab');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

%% ===== Data sources =====
% LaSOT head-tail-40
lasot_sources = {
    struct('label', 'MyECOTracker', ...
           'summary',  fullfile(repo_root, 'jetson_reports', 'verified_otb936_dual_acc_lasot_headtail40', 'summary.csv'), ...
           'per_seq',  fullfile(repo_root, 'jetson_reports', 'verified_otb936_dual_acc_lasot_headtail40', 'per_sequence_metrics.csv')), ...
    struct('label', 'OSTrack-384', ...
           'summary',  fullfile(repo_root, 'OtherTracker', 'lasot', 'lasot936', 'OSTrack', 'summary.csv'), ...
           'per_seq',  fullfile(repo_root, 'OtherTracker', 'lasot', 'lasot936', 'OSTrack', 'per_sequence_metrics.csv')), ...
    struct('label', 'STARK-ST101', ...
           'summary',  fullfile(repo_root, 'OtherTracker', 'lasot', 'lasot936', 'STARK', 'summary.csv'), ...
           'per_seq',  fullfile(repo_root, 'OtherTracker', 'lasot', 'lasot936', 'STARK', 'per_sequence_metrics.csv')) ...
};

% OTB-100
otb_sources = {
    struct('label', 'MyECOTracker', ...
           'summary',  fullfile(repo_root, 'overall_result', 'jetson_verified_otb936_dual_acc_otb100_full_20260404_1618', 'summary.csv'), ...
           'per_seq',  fullfile(repo_root, 'overall_result', 'jetson_verified_otb936_dual_acc_otb100_full_20260404_1618', 'per_sequence_metrics.csv')), ...
    struct('label', 'OSTrack-384', ...
           'summary',  fullfile(repo_root, 'OtherTracker', 'OSTrack', 'otb100_results', 'summary.csv'), ...
           'per_seq',  fullfile(repo_root, 'OtherTracker', 'OSTrack', 'otb100_results', 'per_sequence_metrics.csv')), ...
    struct('label', 'STARK-ST101', ...
           'summary',  fullfile(repo_root, 'OtherTracker', 'Stark', 'otb100_results', 'summary.csv'), ...
           'per_seq',  fullfile(repo_root, 'OtherTracker', 'Stark', 'otb100_results', 'per_sequence_metrics.csv')) ...
};

colors = [0.20 0.40 0.85;   % MyECOTracker - blue
          0.85 0.25 0.20;   % OSTrack      - red
          0.20 0.65 0.30];  % STARK        - green

%% ===== LaSOT plots =====
fprintf('=== LaSOT head-tail-40 ===\n');
lasot_data = load_all(lasot_sources);
plot_summary_bars(lasot_data, colors, 'LaSOT', ...
                  fullfile(out_dir, 'lasot_headtail40_summary'));
plot_per_sequence_auc(lasot_data, colors, 'LaSOT', ...
                      fullfile(out_dir, 'lasot_headtail40_per_sequence_auc'));
plot_auc_fps_scatter(lasot_data, colors, 'LaSOT', ...
                     fullfile(out_dir, 'lasot_headtail40_auc_vs_fps'));

%% ===== OTB plots =====
fprintf('\n=== OTB-100 ===\n');
otb_data = load_all(otb_sources);
plot_summary_bars(otb_data, colors, 'OTB-100', ...
                  fullfile(out_dir, 'otb100_summary'));
plot_per_sequence_auc(otb_data, colors, 'OTB-100', ...
                      fullfile(out_dir, 'otb100_per_sequence_auc'));
plot_auc_fps_scatter(otb_data, colors, 'OTB-100', ...
                     fullfile(out_dir, 'otb100_auc_vs_fps'));

fprintf('\nFigures saved to: %s\n', out_dir);
end


%% ===== Helpers =====

function data = load_all(sources)
data = cell(numel(sources), 1);
for i = 1:numel(sources)
    s = sources{i};
    if ~exist(s.summary, 'file')
        fprintf('  [missing] %s -> %s\n', s.label, s.summary);
        data{i} = struct('label', s.label, 'auc', NaN, 'precision', NaN, ...
                         'success50', NaN, 'fps', NaN, ...
                         'seq_names', {{}}, 'seq_auc', [], 'seq_fps', []);
        continue;
    end
    summary = read_summary_csv(s.summary);
    per_seq = read_per_sequence_csv(s.per_seq);
    fprintf('  [ok] %s: AUC=%.2f  P20=%.2f  FPS=%.2f\n', ...
            s.label, summary.auc, summary.precision, summary.fps);
    data{i} = struct('label', s.label, ...
                     'auc', summary.auc, ...
                     'precision', summary.precision, ...
                     'success50', summary.success50, ...
                     'fps', summary.fps, ...
                     'seq_names', {per_seq.names}, ...
                     'seq_auc', per_seq.auc, ...
                     'seq_fps', per_seq.fps);
end
end


function out = read_summary_csv(path)
% Robust CSV row reader that accepts either {AUC,Precision,Success50,FPS_avg_seq}
% (new format) or {AUC_mean,Precision_mean,Success50_mean,FPS_avg_seq} (legacy).
t = readtable(path);
out.auc       = pick_col(t, {'AUC', 'AUC_mean', 'auc'});
out.precision = pick_col(t, {'Precision', 'Precision_mean', 'precision'});
out.success50 = pick_col(t, {'Success50', 'Success50_mean', 'success50'});
out.fps       = pick_col(t, {'FPS_avg_seq', 'fps_avg_seq'});
end


function out = read_per_sequence_csv(path)
out.names = {};
out.auc = [];
out.fps = [];
if ~exist(path, 'file')
    return;
end
t = readtable(path);
name_col = pick_col_text(t, {'sequence', 'name'});
auc_col  = pick_col(t, {'auc', 'AUC', 'AUC_mean'});
fps_col  = pick_col(t, {'fps', 'FPS_avg_seq'});
out.names = name_col;
out.auc = auc_col;
out.fps = fps_col;
end


function v = pick_col(t, names)
v = NaN;
for k = 1:numel(names)
    if any(strcmp(t.Properties.VariableNames, names{k}))
        raw = t.(names{k});
        if iscell(raw)
            raw = str2double(raw);
        end
        v = raw;
        return;
    end
end
end


function c = pick_col_text(t, names)
c = {};
for k = 1:numel(names)
    if any(strcmp(t.Properties.VariableNames, names{k}))
        c = t.(names{k});
        if ~iscell(c)
            c = cellstr(string(c));
        end
        return;
    end
end
end


function plot_summary_bars(data, colors, title_str, out_base)
labels = cellfun(@(d) d.label, data, 'UniformOutput', false);
n = numel(data);
auc       = arrayfun(@(i) data{i}.auc,       1:n);
prec      = arrayfun(@(i) data{i}.precision, 1:n);
succ50    = arrayfun(@(i) data{i}.success50, 1:n);
fps       = arrayfun(@(i) data{i}.fps,       1:n);

fig = figure('Position', [100 100 1200 700], 'Color', 'w', 'Visible', 'off');

subplot(2,2,1);
draw_bar(auc, labels, colors(1:n,:), 'AUC (%)');
title(sprintf('%s — Success AUC', title_str), 'Interpreter', 'none', 'FontWeight','bold');

subplot(2,2,2);
draw_bar(prec, labels, colors(1:n,:), 'Precision @ 20px (%)');
title(sprintf('%s — Precision', title_str), 'Interpreter', 'none', 'FontWeight','bold');

subplot(2,2,3);
draw_bar(succ50, labels, colors(1:n,:), 'Success @ IoU 0.5 (%)');
title(sprintf('%s — Success50', title_str), 'Interpreter', 'none', 'FontWeight','bold');

subplot(2,2,4);
draw_bar(fps, labels, colors(1:n,:), 'FPS (avg / sequence)');
title(sprintf('%s — FPS', title_str), 'Interpreter', 'none', 'FontWeight','bold');

exportgraphics(fig, [out_base '.png'], 'Resolution', 150);
savefig(fig, [out_base '.fig']);
close(fig);
fprintf('  saved: %s.png\n', out_base);
end


function draw_bar(values, labels, colors, ylabel_str)
b = bar(values, 'FaceColor', 'flat');
b.CData = colors;
xticklabels(labels);
ylabel(ylabel_str);
grid on;
ymax = max(values) * 1.15;
if ~isfinite(ymax) || ymax <= 0
    ymax = 1;
end
ylim([0, ymax]);
for i = 1:numel(values)
    if isfinite(values(i))
        text(i, values(i) + ymax * 0.02, sprintf('%.2f', values(i)), ...
             'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');
    end
end
set(gca, 'FontSize', 10);
end


function plot_per_sequence_auc(data, colors, title_str, out_base)
n = numel(data);
% Build union of sequence names from all trackers (ordered by first appearance).
all_names = {};
for i = 1:n
    if isempty(data{i}.seq_names)
        continue;
    end
    for k = 1:numel(data{i}.seq_names)
        nm = data{i}.seq_names{k};
        if ~any(strcmp(all_names, nm))
            all_names{end+1} = nm; %#ok<AGROW>
        end
    end
end
if isempty(all_names)
    return;
end

M = NaN(numel(all_names), n);
for j = 1:n
    nms = data{j}.seq_names;
    aucs = data{j}.seq_auc;
    for k = 1:numel(nms)
        idx = find(strcmp(all_names, nms{k}), 1);
        if ~isempty(idx)
            M(idx, j) = aucs(k);
        end
    end
end

fig = figure('Position', [100 100 max(1600, 24*numel(all_names)) 600], 'Color', 'w', 'Visible', 'off');
b = bar(M, 'grouped');
for j = 1:n
    b(j).FaceColor = colors(j,:);
end
xticks(1:numel(all_names));
xticklabels(all_names);
xtickangle(60);
ylabel('AUC (%)');
ylim([0, 100]);
grid on;
legend(arrayfun(@(i) data{i}.label, 1:n, 'UniformOutput', false), 'Location', 'best');
title(sprintf('%s — per-sequence Success AUC', title_str), 'Interpreter', 'none', 'FontWeight','bold');
set(gca, 'FontSize', 9);

exportgraphics(fig, [out_base '.png'], 'Resolution', 150);
savefig(fig, [out_base '.fig']);
close(fig);
fprintf('  saved: %s.png\n', out_base);
end


function plot_auc_fps_scatter(data, colors, title_str, out_base)
n = numel(data);
fig = figure('Position', [100 100 800 600], 'Color', 'w', 'Visible', 'off');
hold on;
auc = arrayfun(@(i) data{i}.auc, 1:n);
fps = arrayfun(@(i) data{i}.fps, 1:n);
labels = arrayfun(@(i) data{i}.label, 1:n, 'UniformOutput', false);
for i = 1:n
    scatter(fps(i), auc(i), 250, colors(i,:), 'filled', 'MarkerEdgeColor','k');
    text(fps(i) + max(fps)*0.02, auc(i), labels{i}, 'FontSize', 11, 'FontWeight','bold');
end
xlabel('FPS (avg / sequence)');
ylabel('AUC (%)');
title(sprintf('%s — Accuracy vs Speed', title_str), 'Interpreter', 'none', 'FontWeight','bold');
grid on;
xlim([0, max(fps) * 1.3]);
ylim([0, 100]);
set(gca, 'FontSize', 11);

exportgraphics(fig, [out_base '.png'], 'Resolution', 150);
savefig(fig, [out_base '.fig']);
close(fig);
fprintf('  saved: %s.png\n', out_base);
end
