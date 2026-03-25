function summary = benchmark_otb_tracker(opts)
%BENCHMARK_OTB_TRACKER Run an OTB benchmark for a MATLAB tracker.
%
% Required fields:
%   tracker_name, tracker_path, main_func, otb_root, sequence_file, out_dir
%
% Optional fields:
%   setup_func, toolkit_path, start_index, end_index, resume

opts = normalize_options(opts);
ensure_toolkit_paths(opts.toolkit_path);

sequences = load_otb_sequences(opts.otb_root, opts.sequence_file);
num_sequences = numel(sequences);

if num_sequences == 0
    error('benchmark_otb_tracker:noSequences', 'No OTB sequences found.');
end

mkdir_if_needed(opts.out_dir);

start_index = max(1, opts.start_index);
end_index = min(num_sequences, opts.end_index);

for iseq = start_index:end_index
    seq = sequences{iseq};
    result_file = fullfile(opts.out_dir, sprintf('%s_%s.mat', seq.name, opts.tracker_name));

    if opts.resume && result_has_fps(result_file)
        fprintf('[skip] %3d/%3d %s\n', iseq, num_sequences, seq.name);
        continue;
    end

    fprintf('[run ] %3d/%3d %s -> %s\n', iseq, num_sequences, seq.name, opts.tracker_name);
    [subSeqs, ~] = splitSeqTRE(seq, 1, seq.annos);
    sub_seq = subSeqs{1};

    old_dir = pwd;
    old_path = path;
    try
        cd(opts.tracker_path);
        if ~isempty(opts.setup_func)
            feval(opts.setup_func);
        end
        res = feval(opts.main_func, sub_seq, opts.out_dir, false);
        path(old_path);
        cd(old_dir);
    catch err
        path(old_path);
        cd(old_dir);
        save_error(opts.out_dir, opts.tracker_name, seq.name, err);
        fprintf(2, '[fail] %s\n%s\n', seq.name, getReport(err, 'extended', 'hyperlinks', 'off'));
        continue;
    end

    res.len = sub_seq.len;
    res.annoBegin = sub_seq.annoBegin;
    res.startFrame = sub_seq.startFrame;
    results = {res}; %#ok<NASGU>
    save(result_file, 'results');
end

[rows, summary] = collect_results(sequences, opts.tracker_name, opts.out_dir);
write_per_sequence_csv(rows, fullfile(opts.out_dir, sprintf('%s_per_sequence.csv', lower(opts.tracker_name))));
write_summary_csv(summary, fullfile(opts.out_dir, sprintf('%s_summary.csv', lower(opts.tracker_name))));

end

function opts = normalize_options(opts)
required_fields = {'tracker_name', 'tracker_path', 'main_func', 'otb_root', 'sequence_file', 'out_dir'};
for i = 1:numel(required_fields)
    field_name = required_fields{i};
    if ~isfield(opts, field_name) || isempty(opts.(field_name))
        error('benchmark_otb_tracker:missingOption', 'Missing required option: %s', field_name);
    end
end

if ~isfield(opts, 'setup_func')
    opts.setup_func = '';
end

if ~isfield(opts, 'toolkit_path') || isempty(opts.toolkit_path)
    repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
    opts.toolkit_path = fullfile(repo_root, 'otb', 'otb-toolkit');
end

if ~isfield(opts, 'start_index') || isempty(opts.start_index)
    opts.start_index = 1;
end

if ~isfield(opts, 'end_index') || isempty(opts.end_index)
    opts.end_index = inf;
end

if ~isfield(opts, 'resume') || isempty(opts.resume)
    opts.resume = true;
end
end

function ensure_toolkit_paths(toolkit_path)
addpath(fullfile(toolkit_path, 'utils'));
addpath(fullfile(toolkit_path, 'configs'));
addpath(fullfile(toolkit_path, 'evals'));
addpath(fullfile(toolkit_path, 'tracker_benchmark_v1.0', 'util'));
addpath(fullfile(toolkit_path, 'tracker_benchmark_v1.0', 'rstEval'));
set_global_variable('toolkit_path', toolkit_path);
mkdir_if_needed(fullfile(toolkit_path, 'cache'));
end

function tf = result_has_fps(result_file)
tf = false;
if ~exist(result_file, 'file')
    return;
end

loaded = load(result_file);
if ~isfield(loaded, 'results') || isempty(loaded.results)
    return;
end

first = loaded.results{1};
tf = isfield(first, 'fps') && ~isempty(first.fps) && isfinite(first.fps) && first.fps > 0;
end

function save_error(out_dir, tracker_name, sequence_name, err)
error_file = fullfile(out_dir, sprintf('%s_%s_error.mat', sequence_name, tracker_name));
save(error_file, 'err');
end

function [rows, summary] = collect_results(sequences, tracker_name, out_dir)
rows = struct('sequence', {}, 'len', {}, 'fps', {}, 'total_time_sec', {}, 'status', {});
fps_values = [];
lengths = [];
total_time_values = [];

for i = 1:numel(sequences)
    seq = sequences{i};
    result_file = fullfile(out_dir, sprintf('%s_%s.mat', seq.name, tracker_name));
    row = struct('sequence', seq.name, 'len', NaN, 'fps', NaN, 'total_time_sec', NaN, 'status', 'missing');

    if exist(result_file, 'file')
        loaded = load(result_file);
        if isfield(loaded, 'results') && ~isempty(loaded.results)
            first = loaded.results{1};
            if isfield(first, 'len')
                row.len = double(first.len);
            end
            if isfield(first, 'fps')
                row.fps = double(first.fps);
            end
            if isfinite(row.len) && isfinite(row.fps) && row.fps > 0
                row.total_time_sec = row.len / row.fps;
                row.status = 'ok';
                fps_values(end + 1) = row.fps; %#ok<AGROW>
                lengths(end + 1) = row.len; %#ok<AGROW>
                total_time_values(end + 1) = row.total_time_sec; %#ok<AGROW>
            else
                row.status = 'invalid';
            end
        end
    end

    rows(end + 1) = row; %#ok<AGROW>
end

summary = struct();
summary.tracker = tracker_name;
summary.source = 'local_matlab_otb';
summary.valid_sequences = numel(fps_values);
summary.total_frames = sum(lengths);
summary.total_time_sec = sum(total_time_values);

if summary.valid_sequences == numel(sequences)
    summary.status = 'ok';
elseif summary.valid_sequences > 0
    summary.status = 'partial';
else
    summary.status = 'missing';
end

if isempty(fps_values)
    summary.fps_avg_seq = NaN;
    summary.fps_median_seq = NaN;
    summary.fps_global = NaN;
    summary.fps_weighted_by_frames = NaN;
else
    summary.fps_avg_seq = mean(fps_values);
    summary.fps_median_seq = median(fps_values);
    summary.fps_global = summary.fps_avg_seq;
    summary.fps_weighted_by_frames = summary.total_frames / summary.total_time_sec;
end
end

function write_per_sequence_csv(rows, csv_path)
fid = fopen(csv_path, 'w');
if fid == -1
    error('benchmark_otb_tracker:writeFailed', 'Failed to write CSV: %s', csv_path);
end
cleanup_obj = onCleanup(@() fclose(fid)); %#ok<NASGU>

fprintf(fid, 'sequence,len,fps,total_time_sec,status\n');
for i = 1:numel(rows)
    row = rows(i);
    fprintf(fid, '%s,%s,%s,%s,%s\n', ...
        row.sequence, num_to_str(row.len), num_to_str(row.fps), ...
        num_to_str(row.total_time_sec), row.status);
end
end

function write_summary_csv(summary, csv_path)
fid = fopen(csv_path, 'w');
if fid == -1
    error('benchmark_otb_tracker:writeFailed', 'Failed to write CSV: %s', csv_path);
end
cleanup_obj = onCleanup(@() fclose(fid)); %#ok<NASGU>

fprintf(fid, 'tracker,source,status,valid_sequences,fps_avg_seq,fps_median_seq,fps_global,fps_weighted_by_frames,total_frames,total_time_sec\n');
fprintf(fid, '%s,%s,%s,%d,%s,%s,%s,%s,%s,%s\n', ...
    summary.tracker, summary.source, summary.status, summary.valid_sequences, ...
    num_to_str(summary.fps_avg_seq), num_to_str(summary.fps_median_seq), ...
    num_to_str(summary.fps_global), num_to_str(summary.fps_weighted_by_frames), ...
    num_to_str(summary.total_frames), ...
    num_to_str(summary.total_time_sec));
end

function value = num_to_str(number)
if ~isfinite(number)
    value = 'nan';
else
    value = sprintf('%.15g', number);
end
end

function mkdir_if_needed(path_str)
if ~exist(path_str, 'dir')
    mkdir(path_str);
end
end
