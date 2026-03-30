function summary = run_meem_lasot_headtail40(lasot_root, sequence_file, out_dir, start_index, end_index)
%RUN_MEEM_LASOT_HEADTAIL40 Run MEEM on the LaSOT head/tail 40 subset.

repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));

if nargin < 1 || isempty(lasot_root)
    lasot_root = fullfile(repo_root, 'ls', 'lasot');
end

if nargin < 2 || isempty(sequence_file)
    sequence_file = fullfile(repo_root, 'OtherTracker', 'lasot', 'lasot936', 'headtail40_sequences.txt');
end

if nargin < 3 || isempty(out_dir)
    out_dir = fullfile(repo_root, 'OtherTracker', 'lasot', 'lasot936', 'MEEM');
end

if nargin < 4 || isempty(start_index)
    start_index = 1;
end

if nargin < 5 || isempty(end_index)
    end_index = inf;
end

tracker_root = fileparts(mfilename('fullpath'));
result_dir = fullfile(out_dir, 'tracking_results', 'MEEM');
mkdir_if_needed(out_dir);
mkdir_if_needed(result_dir);

sequence_names = read_sequence_names(sequence_file);
num_sequences = numel(sequence_names);

start_index = max(1, start_index);
end_index = min(num_sequences, end_index);

fps_values = [];
frame_counts = [];
time_totals = [];

for idx = start_index:end_index
    seq_name = sequence_names{idx};
    bbox_file = fullfile(result_dir, [seq_name '.txt']);
    time_file = fullfile(result_dir, [seq_name '_time.txt']);

    if exist(bbox_file, 'file') && exist(time_file, 'file')
        [seq_frames, seq_fps, seq_time] = summarize_existing_result(time_file);
        fprintf('[skip] %3d/%3d %s fps=%.6f\n', idx, num_sequences, seq_name, seq_fps);
        fps_values(end + 1) = seq_fps; %#ok<AGROW>
        frame_counts(end + 1) = seq_frames; %#ok<AGROW>
        time_totals(end + 1) = seq_time; %#ok<AGROW>
        continue;
    end

    seq = build_lasot_sequence(lasot_root, seq_name);
    fprintf('[run ] %3d/%3d %s\n', idx, num_sequences, seq_name);

    old_dir = pwd;
    old_path = path;
    try
        cd(tracker_root);
        setup_paths();
        res = run_meem(seq, '', false);
        path(old_path);
        cd(old_dir);
    catch err
        path(old_path);
        cd(old_dir);
        rethrow(err);
    end

    tracked_bb = double(res.res);
    dlmwrite(bbox_file, tracked_bb, 'delimiter', ',', 'precision', '%.6f');

    seq_fps = double(res.fps);
    seq_time = seq.len / seq_fps;
    per_frame_time = repmat(seq_time / seq.len, seq.len, 1);
    dlmwrite(time_file, per_frame_time, 'delimiter', '\t', 'precision', '%.10f');

    fprintf('[done] %3d/%3d %s fps=%.6f\n', idx, num_sequences, seq_name, seq_fps);

    fps_values(end + 1) = seq_fps; %#ok<AGROW>
    frame_counts(end + 1) = seq.len; %#ok<AGROW>
    time_totals(end + 1) = seq_time; %#ok<AGROW>
end

summary = struct();
summary.tracker = 'MEEM';
summary.scope = 'headtail40';
summary.valid_sequences = numel(fps_values);
summary.fps_avg_seq = mean(fps_values);
summary.fps_median_seq = median(fps_values);
summary.total_frames = sum(frame_counts);
summary.total_time_sec = sum(time_totals);
summary.fps_weighted_by_frames = summary.total_frames / summary.total_time_sec;

write_summary_csv(fullfile(out_dir, 'meem_matlab_summary.csv'), summary);
end

function seq = build_lasot_sequence(lasot_root, seq_name)
parts = regexp(seq_name, '^([^-]+)-', 'tokens', 'once');
if isempty(parts)
    error('run_meem_lasot_headtail40:invalidSequence', 'Invalid sequence name: %s', seq_name);
end

class_name = parts{1};
seq_dir = fullfile(lasot_root, class_name, seq_name);
gt_path = fullfile(seq_dir, 'groundtruth.txt');
img_dir = fullfile(seq_dir, 'img');

if ~exist(gt_path, 'file')
    error('run_meem_lasot_headtail40:missingGt', 'Missing groundtruth for %s', seq_name);
end

gt = dlmread(gt_path);
seq = struct();
seq.name = seq_name;
seq.len = size(gt, 1);
seq.init_rect = gt(1, :);
frames = cell(seq.len, 1);
for frame_idx = 1:seq.len
    frames{frame_idx} = fullfile(img_dir, sprintf('%08d.jpg', frame_idx));
end
seq.s_frames = frames;
end

function [num_frames, fps_value, total_time] = summarize_existing_result(time_file)
times = dlmread(time_file);
times = times(:);
num_frames = numel(times);
total_time = sum(times);
fps_value = num_frames / total_time;
end

function names = read_sequence_names(sequence_file)
fid = fopen(sequence_file, 'r');
if fid == -1
    error('run_meem_lasot_headtail40:sequenceFile', 'Could not open sequence file: %s', sequence_file);
end
cleanup_obj = onCleanup(@() fclose(fid)); %#ok<NASGU>

names = {};
while ~feof(fid)
    line = strtrim(fgetl(fid));
    if ischar(line) && ~isempty(line)
        names{end + 1} = line; %#ok<AGROW>
    end
end
end

function write_summary_csv(csv_path, summary)
fid = fopen(csv_path, 'w');
if fid == -1
    error('run_meem_lasot_headtail40:summaryWrite', 'Could not write summary CSV: %s', csv_path);
end
cleanup_obj = onCleanup(@() fclose(fid)); %#ok<NASGU>

fprintf(fid, 'tracker,scope,valid_sequences,fps_avg_seq,fps_median_seq,fps_weighted_by_frames,total_frames,total_time_sec\n');
fprintf(fid, '%s,%s,%d,%.15g,%.15g,%.15g,%.15g,%.15g\n', ...
    summary.tracker, summary.scope, summary.valid_sequences, summary.fps_avg_seq, ...
    summary.fps_median_seq, summary.fps_weighted_by_frames, summary.total_frames, summary.total_time_sec);
end

function mkdir_if_needed(path_str)
if ~exist(path_str, 'dir')
    mkdir(path_str);
end
end
