function summary = run_ccot_otb(otb_root, sequence_file, out_dir, start_index, end_index)
%RUN_CCOT_OTB Benchmark CCOT on OTB100 using the ECCV 2016 settings.

repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
tools_path = fullfile(repo_root, 'OtherTracker', 'tools');
addpath(tools_path);

if nargin < 1 || isempty(otb_root)
    otb_root = fullfile(repo_root, 'otb', 'otb100');
end

if nargin < 2 || isempty(sequence_file)
    sequence_file = fullfile(repo_root, 'otb', 'otb-toolkit', 'sequences', 'SEQUENCES');
end

if nargin < 3 || isempty(out_dir)
    out_dir = fullfile(repo_root, 'OtherTracker', 'CCOT', 'otb100_fps');
end

if nargin < 4 || isempty(start_index)
    start_index = 1;
end

if nargin < 5 || isempty(end_index)
    end_index = inf;
end

opts = struct();
opts.tracker_name = 'CCOT';
opts.tracker_path = fileparts(mfilename('fullpath'));
opts.setup_func = 'setup_paths';
opts.main_func = 'ECCV2016_settings';
opts.otb_root = otb_root;
opts.sequence_file = sequence_file;
opts.out_dir = out_dir;
opts.start_index = start_index;
opts.end_index = end_index;
opts.resume = true;

summary = benchmark_otb_tracker(opts);
end
