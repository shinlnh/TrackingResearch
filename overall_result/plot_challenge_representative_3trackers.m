function plot_challenge_representative_3trackers()
% PLOT_CHALLENGE_REPRESENTATIVE_3TRACKERS
% Reproduces the "challenge-representative" comparison plots (IoU per frame
% + trajectory overlay) but with three trackers: MyECOTracker, OSTrack, STARK.
%
% Produces two figures (one per dataset):
%   - otb100_challenge_representative_3trackers.png
%   - lasot_challenge_representative_3trackers.png
%
% Layout: 3 rows (challenge categories) x 2 columns (IoU plot, trajectory).
%
% Output dir: overall_result/sota_comparison/matlab/

repo_root = fileparts(fileparts(mfilename('fullpath')));
out_dir = fullfile(repo_root, 'overall_result', 'sota_comparison', 'matlab');
if ~exist(out_dir, 'dir')
    mkdir(out_dir);
end

%% ===== Tracker color scheme =====
% MyECO blue, OSTrack red, STARK green — matches the summary bar plots.
tracker_colors = struct( ...
    'MyECOTracker', [0.20 0.40 0.85], ...
    'OSTrack',      [0.85 0.25 0.20], ...
    'STARK',        [0.20 0.65 0.30]);

%% ===== OTB-100: 3 challenge representatives =====
otb_specs = {
    struct('challenge', 'Occlusion (OCC)', 'seq', 'Bolt'), ...
    struct('challenge', 'Illumination Variation (IV)', 'seq', 'Human8'), ...
    struct('challenge', 'Fast Motion (FM)', 'seq', 'BlurCar3') ...
};

otb_paths = struct( ...
    'MyECOTracker', fullfile(repo_root, 'overall_result', ...
        'jetson_verified_otb936_dual_acc_otb100_full_20260404_1618', ...
        'tracking_results', 'verified_otb936_jetson_fast_trt_dual_acc_981'), ...
    'OSTrack', fullfile(repo_root, 'OtherTracker', 'OSTrack', ...
        'otb100_results', 'tracking_results', 'OSTrack'), ...
    'STARK', fullfile(repo_root, 'OtherTracker', 'Stark', ...
        'otb100_results', 'tracking_results', 'STARK'));

otb_dataset_root = fullfile(repo_root, 'otb', 'otb100');

fprintf('=== OTB-100 challenge plots ===\n');
plot_dataset(otb_specs, otb_paths, otb_dataset_root, 'otb', ...
             tracker_colors, ...
             'MyTracker vs OSTrack vs STARK on Representative OTB Challenge Sequences', ...
             fullfile(out_dir, 'otb100_challenge_representative_3trackers'));

%% ===== LaSOT head-tail-40: 3 challenge representatives =====
lasot_specs = {
    struct('challenge', 'Occlusion (POC/FOC)', 'seq', 'basketball-7'), ...
    struct('challenge', 'Illumination Variation (IV)', 'seq', 'shark-5'), ...
    struct('challenge', 'Fast Motion (FM)', 'seq', 'airplane-13') ...
};

lasot_paths = struct( ...
    'MyECOTracker', fullfile(repo_root, 'MyECOTracker', 'pytracking', ...
        'pytracking', 'tracking_results', 'eco', 'verified_otb936_936'), ...
    'OSTrack', fullfile(repo_root, 'OtherTracker', 'lasot', 'lasot936', ...
        'OSTrack', 'tracking_results', 'OSTrack'), ...
    'STARK', fullfile(repo_root, 'OtherTracker', 'lasot', 'lasot936', ...
        'STARK', 'tracking_results', 'STARK'));

lasot_dataset_root = fullfile(repo_root, 'ls', 'lasot');

fprintf('\n=== LaSOT challenge plots ===\n');
plot_dataset(lasot_specs, lasot_paths, lasot_dataset_root, 'lasot', ...
             tracker_colors, ...
             'MyTracker vs OSTrack vs STARK on Representative LaSOT Challenge Sequences', ...
             fullfile(out_dir, 'lasot_challenge_representative_3trackers'));

fprintf('\nFigures saved to: %s\n', out_dir);
end


%% ====================== Helpers ======================

function plot_dataset(specs, paths, dataset_root, dataset_type, ...
                      tracker_colors, super_title, out_base)
n_rows = numel(specs);
fig = figure('Position', [50 50 2000 550 * n_rows], 'Color', 'w', 'Visible', 'off');
sgtitle(super_title, 'FontSize', 16, 'FontWeight', 'bold', 'Interpreter', 'none');

tracker_names = fieldnames(tracker_colors);  % MyECOTracker, OSTrack, STARK
tracker_labels = struct('MyECOTracker', 'MyTracker', ...
                        'OSTrack', 'OSTrack-384', ...
                        'STARK', 'STARK-ST101');

for r = 1:n_rows
    spec = specs{r};
    seq = spec.seq;
    fprintf('  [%d/%d] %s — %s\n', r, n_rows, spec.challenge, seq);

    [gt, first_img] = load_groundtruth(dataset_root, dataset_type, seq);
    if isempty(gt)
        fprintf('    SKIP (no ground truth)\n');
        continue;
    end

    % Read tracker outputs
    bboxes = struct();
    aucs = struct();
    for t = 1:numel(tracker_names)
        tn = tracker_names{t};
        fpath = fullfile(paths.(tn), [seq '.txt']);
        bbox = load_tracker_bbox(fpath);
        if isempty(bbox)
            fprintf('    %s: missing %s\n', tn, fpath);
            bbox = NaN(size(gt));
        end
        % Align lengths to min
        L = min(size(bbox,1), size(gt,1));
        bbox = bbox(1:L, :);
        bboxes.(tn) = bbox;

        % Compute IoU per frame
        iou_pct = compute_iou_per_frame(bbox, gt(1:L, :)) * 100;
        bboxes.([tn '_iou']) = iou_pct;
        aucs.(tn) = mean(iou_pct, 'omitnan');
    end

    % --- LEFT: IoU per frame ---
    subplot(n_rows, 2, 2*r - 1); hold on;
    for t = 1:numel(tracker_names)
        tn = tracker_names{t};
        iou = bboxes.([tn '_iou']);
        plot(1:numel(iou), iou, '-', 'Color', tracker_colors.(tn), ...
             'LineWidth', 1.1, 'DisplayName', tracker_labels.(tn));
    end
    xlabel('Frame'); ylabel('IoU vs Ground Truth (%)');
    ylim([0 100]); grid on;
    title_str = sprintf('%s - %s\nAUC: My %.1f | OS %.1f | STARK %.1f', ...
                        spec.challenge, seq, ...
                        aucs.MyECOTracker, aucs.OSTrack, aucs.STARK);
    title(title_str, 'Interpreter', 'none', 'FontSize', 10);
    legend('Location', 'best', 'Interpreter', 'none', 'FontSize', 8);
    set(gca, 'FontSize', 9);

    % --- RIGHT: trajectory overlay on first frame ---
    ax = subplot(n_rows, 2, 2*r);
    if ~isempty(first_img)
        imshow(first_img, 'Parent', ax);
        hold(ax, 'on');
        set(ax, 'YDir', 'reverse');
    else
        hold(ax, 'on');
    end
    for t = 1:numel(tracker_names)
        tn = tracker_names{t};
        bbox = bboxes.(tn);
        cx = bbox(:, 1) + 0.5 * bbox(:, 3);
        cy = bbox(:, 2) + 0.5 * bbox(:, 4);
        valid = isfinite(cx) & isfinite(cy);
        plot(cx(valid), cy(valid), '-', ...
             'Color', tracker_colors.(tn), 'LineWidth', 1.2, ...
             'DisplayName', [tracker_labels.(tn) ' trajectory']);
        if any(valid)
            idx_first = find(valid, 1, 'first');
            idx_last = find(valid, 1, 'last');
            plot(cx(idx_first), cy(idx_first), 'o', ...
                 'MarkerFaceColor', tracker_colors.(tn), ...
                 'MarkerEdgeColor', 'k', 'MarkerSize', 6, ...
                 'HandleVisibility', 'off');
            plot(cx(idx_last), cy(idx_last), 'x', ...
                 'Color', tracker_colors.(tn), 'MarkerSize', 9, ...
                 'LineWidth', 2, 'HandleVisibility', 'off');
        end
    end
    title(ax, sprintf('Bounding-box Center Trajectory by Frame - %s', seq), ...
          'Interpreter', 'none', 'FontSize', 11);
    xlabel(ax, 'x pixel'); ylabel(ax, 'y pixel');
    axis(ax, 'on');                 % show axis box + ticks
    set(ax, 'XColor', [0 0 0], 'YColor', [0 0 0]);
    if r == 1
        legend(ax, 'Location', 'best', 'Interpreter', 'none', 'FontSize', 9);
    end
    set(ax, 'FontSize', 10);
end

exportgraphics(fig, [out_base '.png'], 'Resolution', 150);
savefig(fig, [out_base '.fig']);
close(fig);
fprintf('  saved: %s.png\n', out_base);
end


function bbox = load_tracker_bbox(path)
% Load a tracker bbox file. Auto-detects comma/tab/space separator.
bbox = [];
if ~exist(path, 'file')
    return;
end
try
    raw = fileread(path);
    if isempty(strtrim(raw))
        return;
    end
    % Replace tabs with commas for uniform parsing
    raw = strrep(raw, char(9), ',');
    raw = strrep(raw, ' ', ',');
    lines = regexp(raw, '\r?\n', 'split');
    rows = [];
    for k = 1:numel(lines)
        line = strtrim(lines{k});
        if isempty(line)
            continue;
        end
        % Collapse repeated commas
        line = regexprep(line, ',+', ',');
        nums = sscanf(line, '%f,%f,%f,%f');
        if numel(nums) >= 4
            rows(end+1, :) = nums(1:4)'; %#ok<AGROW>
        end
    end
    bbox = rows;
catch ME
    warning('Failed to read %s: %s', path, ME.message);
end
end


function [gt, first_img] = load_groundtruth(dataset_root, dataset_type, seq)
gt = []; first_img = [];

% OTB special-cased sequences (handled inline)
otb_special = containers.Map( ...
    {'Board', 'David', 'Football1', 'Freeman3', 'Freeman4', ...
     'BlurCar1', 'BlurCar3', 'BlurCar4', 'Tiger1'}, ...
    {[1, 698, 5], [300, 770, 4], [1, 74, 4], [1, 460, 4], [1, 283, 4], ...
     [247, 988, 4], [3, 359, 4], [18, 397, 4], [6, 354, 4]});

if strcmpi(dataset_type, 'otb')
    base_dir = fullfile(dataset_root, seq);
    gt_path = fullfile(base_dir, 'groundtruth_rect.txt');
    img_dir = fullfile(base_dir, 'img');
    if isKey(otb_special, seq)
        info = otb_special(seq);
        start_f = info(1); nz = info(3);
    else
        start_f = 1; nz = 4;
    end
    first_img_path = fullfile(img_dir, sprintf(['%0' num2str(nz) 'd.jpg'], start_f));
elseif strcmpi(dataset_type, 'lasot')
    parts = regexp(seq, '-', 'split');
    category = parts{1};
    base_dir = fullfile(dataset_root, category, seq);
    gt_path = fullfile(base_dir, 'groundtruth.txt');
    img_dir = fullfile(base_dir, 'img');
    first_img_path = fullfile(img_dir, '00000001.jpg');
else
    return;
end

if ~exist(gt_path, 'file')
    fprintf('    GT missing: %s\n', gt_path);
    return;
end
gt = load_tracker_bbox(gt_path);

if exist(first_img_path, 'file')
    try
        first_img = imread(first_img_path);
    catch
        first_img = [];
    end
end
end


function iou = compute_iou_per_frame(boxes_a, boxes_b)
% boxes_a, boxes_b: Nx4 in [x, y, w, h]
n = size(boxes_a, 1);
iou = zeros(n, 1);
for i = 1:n
    a = boxes_a(i, :); b = boxes_b(i, :);
    if any(~isfinite(a)) || any(~isfinite(b)) || a(3) <= 0 || a(4) <= 0 || b(3) <= 0 || b(4) <= 0
        iou(i) = 0;
        continue;
    end
    ax2 = a(1) + a(3); ay2 = a(2) + a(4);
    bx2 = b(1) + b(3); by2 = b(2) + b(4);
    ix1 = max(a(1), b(1)); iy1 = max(a(2), b(2));
    ix2 = min(ax2, bx2);   iy2 = min(ay2, by2);
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1);
    inter = iw * ih;
    union = a(3)*a(4) + b(3)*b(4) - inter;
    if union > 0
        iou(i) = inter / union;
    end
end
end
