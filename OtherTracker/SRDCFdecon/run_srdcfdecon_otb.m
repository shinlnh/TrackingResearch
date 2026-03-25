function summary = run_srdcfdecon_otb(otb_root, sequence_file, out_dir, start_index, end_index)
%RUN_SRDCFDECON_OTB Benchmark SRDCFdecon on OTB using the CVPR 2016 settings.

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
    out_dir = fullfile(repo_root, 'OtherTracker', 'SRDCFdecon', 'otb100_fps');
end

if nargin < 4 || isempty(start_index)
    start_index = 1;
end

if nargin < 5 || isempty(end_index)
    end_index = inf;
end

opts = struct();
opts.tracker_name = 'SRDCFdecon';
opts.tracker_path = fullfile(fileparts(mfilename('fullpath')), 'SRDCFdecon');
opts.setup_func = 'setup_paths';
opts.main_func = 'OTB_settings';
opts.otb_root = otb_root;
opts.sequence_file = sequence_file;
opts.out_dir = out_dir;
opts.start_index = start_index;
opts.end_index = end_index;
opts.resume = true;

summary = benchmark_otb_tracker(opts);
end
