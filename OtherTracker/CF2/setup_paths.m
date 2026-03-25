function setup_paths()
%SETUP_PATHS Prepare CF2 paths and MatConvNet for local benchmarking.

root_path = fileparts(mfilename('fullpath'));
addpath(root_path);
addpath(fullfile(root_path, 'utility'));
addpath(fullfile(root_path, 'model'));
addpath(fullfile(root_path, 'cf_scale'));

current_path = getenv('PATH');
path_entries = {
    root_path
    fullfile(root_path, 'utility')
    fullfile(root_path, 'cf_scale')
    fullfile(root_path, 'external', 'matconvnet', 'matlab', 'mex')
    };

for i = 1:numel(path_entries)
    entry = path_entries{i};
    if exist(entry, 'dir') && isempty(strfind(lower(current_path), lower(entry))) %#ok<STREMP>
        current_path = [entry pathsep current_path];
    end
end
setenv('PATH', current_path);

matconvnet_path = fullfile(root_path, 'external', 'matconvnet', 'matlab');
addpath(matconvnet_path);
if exist(fullfile(matconvnet_path, 'vl_setupnn.m'), 'file')
    run(fullfile(matconvnet_path, 'vl_setupnn.m'));
end
end
