function summary = run_verified_lasot_headtail40(tracker_name, lasot_root, sequence_file, out_dir, start_index, end_index)
%RUN_VERIFIED_LASOT_HEADTAIL40 Run a verified tracker on the LaSOT head/tail 40 subset.
%
% Supported trackers in this generic driver:
%   ASLA, BACF, CN, CSK, CT, CSRDCF, DSiam, ECO_HC, fDSST, IVT,
%   L1APG, MIL, SCT4, SiamFC, Staple_CA, STC, STRCF, Struck, TLD

repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));

if nargin < 1 || isempty(tracker_name)
    error('run_verified_lasot_headtail40:trackerRequired', 'tracker_name is required');
end

if nargin < 2 || isempty(lasot_root)
    lasot_root = fullfile(repo_root, 'ls', 'lasot');
end

if nargin < 3 || isempty(sequence_file)
    sequence_file = fullfile(repo_root, 'OtherTracker', 'lasot', 'lasot936', 'headtail40_sequences.txt');
end

if nargin < 4 || isempty(out_dir)
    out_dir = fullfile(repo_root, 'OtherTracker', 'lasot', 'lasot936', tracker_name);
end

if nargin < 5 || isempty(start_index)
    start_index = 1;
end

if nargin < 6 || isempty(end_index)
    end_index = inf;
end

cfg = tracker_config(tracker_name);
tracker_root = fullfile(repo_root, 'OtherTracker', 'verified', cfg.root_dir);
if exist(tracker_root, 'dir') ~= 7
    error('run_verified_lasot_headtail40:missingRoot', 'Missing tracker root: %s', tracker_root);
end

result_dir = fullfile(out_dir, 'tracking_results', cfg.result_dir);
mkdir_if_needed(out_dir);
mkdir_if_needed(result_dir);

sequence_names = read_sequence_names(sequence_file);
num_sequences = numel(sequence_names);

start_index = max(1, start_index);
end_index = min(num_sequences, end_index);

fps_values = [];
frame_counts = [];
time_totals = [];

old_dir = pwd;
old_path = path;
cleanup_obj = onCleanup(@() restore_environment(old_dir, old_path)); %#ok<NASGU>

cd(tracker_root);
add_config_paths(tracker_root, cfg);
run_setup_commands(cfg);

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

    [tracked_bb, seq_fps, seq_time] = run_single_sequence(cfg, seq);
    dlmwrite(bbox_file, double(tracked_bb), 'delimiter', ',', 'precision', '%.6f');

    per_frame_time = repmat(seq_time / seq.len, seq.len, 1);
    dlmwrite(time_file, per_frame_time, 'delimiter', '\t', 'precision', '%.10f');

    fprintf('[done] %3d/%3d %s fps=%.6f\n', idx, num_sequences, seq_name, seq_fps);
    fps_values(end + 1) = seq_fps; %#ok<AGROW>
    frame_counts(end + 1) = seq.len; %#ok<AGROW>
    time_totals(end + 1) = seq_time; %#ok<AGROW>
end

summary = struct();
summary.tracker = cfg.tracker_label;
summary.scope = 'headtail40';
summary.valid_sequences = numel(fps_values);
summary.fps_avg_seq = mean(fps_values);
summary.fps_median_seq = median(fps_values);
summary.total_frames = sum(frame_counts);
summary.total_time_sec = sum(time_totals);
summary.fps_weighted_by_frames = summary.total_frames / summary.total_time_sec;

write_summary_csv(fullfile(out_dir, 'summary.csv'), summary);
end

function [tracked_bb, seq_fps, seq_time] = run_single_sequence(cfg, seq)
switch cfg.call_style
    case 'path_name'
        tic;
        tracked_bb = feval(cfg.run_function, seq.seq_dir, seq.name);
        seq_time = toc;
        seq_fps = seq.len / seq_time;
    case 'classdir_name'
        tic;
        tracked_bb = feval(cfg.run_function, seq.class_dir, seq.name);
        seq_time = toc;
        seq_fps = seq.len / seq_time;
    case 'seq_struct'
        result = feval(cfg.run_function, seq.struct_arg, '', false);
        tracked_bb = double(result.res);
        seq_fps = double(result.fps);
        seq_time = seq.len / seq_fps;
    case 'dsiam_seq'
        result = feval(cfg.run_function, seq.struct_arg, [], false, fullfile(pwd, 'DSiam'), 'siamfc', '1res');
        tracked_bb = double(result.res);
        seq_fps = double(result.fps);
        seq_time = seq.len / seq_fps;
    otherwise
        error('run_verified_lasot_headtail40:unsupportedCallStyle', ...
            'Unsupported call_style: %s', cfg.call_style);
end

tracked_bb = double(tracked_bb);
if size(tracked_bb, 1) ~= seq.len
    error('run_verified_lasot_headtail40:frameMismatch', ...
        'Tracker %s returned %d frames for %s, expected %d.', ...
        cfg.tracker_label, size(tracked_bb, 1), seq.name, seq.len);
end
end

function seq = build_lasot_sequence(lasot_root, seq_name)
parts = regexp(seq_name, '^([^-]+)-', 'tokens', 'once');
if isempty(parts)
    error('run_verified_lasot_headtail40:invalidSequence', 'Invalid sequence name: %s', seq_name);
end

class_name = parts{1};
seq_dir = fullfile(lasot_root, class_name, seq_name);
gt_path = fullfile(seq_dir, 'groundtruth.txt');
img_dir = fullfile(seq_dir, 'img');

if ~exist(gt_path, 'file')
    error('run_verified_lasot_headtail40:missingGt', 'Missing groundtruth for %s', seq_name);
end

gt = dlmread(gt_path);
seq = struct();
seq.name = seq_name;
seq.len = size(gt, 1);
seq.init_rect = gt(1, :);
seq.seq_dir = seq_dir;
seq.class_dir = fullfile(lasot_root, class_name);

frames = cell(seq.len, 1);
for frame_idx = 1:seq.len
    frames{frame_idx} = fullfile(img_dir, sprintf('%08d.jpg', frame_idx));
end

seq_struct = struct();
seq_struct.name = seq_name;
seq_struct.len = seq.len;
seq_struct.init_rect = seq.init_rect;
seq_struct.s_frames = frames;
seq.struct_arg = seq_struct;
end

function cfg = tracker_config(tracker_name)
switch lower(tracker_name)
    case 'asla'
        cfg = simple_path_cfg('ASLA', 'ASLA', {'ASLA'}, ...
            'run(fullfile(''ASLA'',''vlfeat'',''toolbox'',''vl_setup''));', 'asla_tracker');
    case 'bacf'
        cfg = simple_path_cfg('BACF', 'BACF', {'BACF'}, '', 'run_BACK_tracking');
    case 'cn'
        cfg = simple_path_cfg('CN', 'CN', {'CN'}, '', 'run_tracker');
    case 'csk'
        cfg = simple_path_cfg('CSK', 'CSK', {'CSK'}, '', 'run_tracker');
    case 'ct'
        cfg = simple_path_cfg('CT', 'CT', {'CT'}, '', 'Runtracker');
    case 'csrdcf'
        cfg = simple_path_cfg('CSRDCF', 'CSRDCF', {'CSRDCF', 'CSRDCF/utils', 'CSRDCF/features', 'CSRDCF/mex'}, '', 'demo_csr');
    case 'dsiam'
        cfg = struct();
        cfg.tracker_label = 'DSiam';
        cfg.root_dir = 'DSiam';
        cfg.result_dir = 'DSiam';
        cfg.addpaths = {'DSiam', 'DSiam/utils', 'DSiam/models', 'DSiam/matconvnet/matlab'};
        cfg.setup_cmd = 'vl_setupnn;';
        cfg.run_function = 'run_DSiam_tracker';
        cfg.call_style = 'dsiam_seq';
    case 'eco_hc'
        cfg = simple_path_cfg('ECO_HC', 'ECO_HC', {'ECO_HC'}, 'setup_paths();', 'demo_ECO_HC_gpu');
    case 'fdsst'
        cfg = simple_path_cfg('fDSST', 'fDSST', {'fDSST'}, '', 'run_tracker');
    case 'ivt'
        cfg = simple_path_cfg('IVT', 'IVT', {'IVT'}, '', 'runtracker');
    case 'l1apg'
        cfg = simple_path_cfg('L1APG', 'L1APG', {'L1APG'}, '', 'L1APG_demo');
    case 'mil'
        cfg = simple_path_cfg('MIL', 'MIL', {'MIL'}, '', 'MIL_main');
    case 'sct4'
        cfg = simple_path_cfg('SCT4', 'SCT4', {'SCT4', 'SCT4/KCF', 'SCT4/strong'}, '', 'run_tracker');
        cfg.genpaths = {'SCT4/PiotrDollarToolbox'};
    case 'siamfc'
        cfg = simple_path_cfg('SiamFC', 'SiamFC', {'SiamFC/tracking', 'SiamFC/util', 'SiamFC/matconvnet/matlab'}, ...
            'vl_setupnn;', 'run_tracker');
    case 'staple_ca'
        cfg = simple_path_cfg('Staple_CA', 'Staple_CA', {'Staple_CA'}, '', 'runTracker');
    case 'stc'
        cfg = simple_path_cfg('STC', 'STC', {'STC'}, '', 'demoSTC');
    case 'strcf'
        cfg = struct();
        cfg.tracker_label = 'STRCF';
        cfg.root_dir = 'STRCF';
        cfg.result_dir = 'STRCF';
        cfg.addpaths = {'STRCF'};
        cfg.setup_cmd = 'setup_paths();';
        cfg.run_function = 'run_STRCF_code';
        cfg.call_style = 'seq_struct';
    case 'struck'
        cfg = simple_path_cfg('Struck', 'Struck', {'Struck'}, '', 'demo_struck');
        cfg.call_style = 'classdir_name';
    case 'tld'
        cfg = simple_path_cfg('TLD', 'TLD', {'TLD', 'TLD/tld', 'TLD/utils_tld', 'TLD/img_tld', 'TLD/mex_tld', 'TLD/bbox_tld'}, '', 'run_TLD_demo');
    otherwise
        error('run_verified_lasot_headtail40:unsupportedTracker', ...
            'Unsupported tracker: %s', tracker_name);
end
end

function cfg = simple_path_cfg(label, root_dir, addpaths, setup_cmd, run_function)
cfg = struct();
cfg.tracker_label = label;
cfg.root_dir = root_dir;
cfg.result_dir = label;
cfg.addpaths = addpaths;
cfg.setup_cmd = setup_cmd;
cfg.run_function = run_function;
cfg.call_style = 'path_name';
end

function add_config_paths(tracker_root, cfg)
addpath(tracker_root);
for i = 1:numel(cfg.addpaths)
    addpath(fullfile(tracker_root, cfg.addpaths{i}));
end
if isfield(cfg, 'genpaths')
    for i = 1:numel(cfg.genpaths)
        addpath(genpath(fullfile(tracker_root, cfg.genpaths{i})));
    end
end
end

function run_setup_commands(cfg)
if isfield(cfg, 'setup_cmd') && ~isempty(cfg.setup_cmd)
    eval(cfg.setup_cmd);
end
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
    error('run_verified_lasot_headtail40:sequenceFile', 'Could not open sequence file: %s', sequence_file);
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
    error('run_verified_lasot_headtail40:summaryWrite', 'Could not write summary CSV: %s', csv_path);
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

function restore_environment(old_dir, old_path)
try
    path(old_path);
catch
end
try
    cd(old_dir);
catch
end
end
