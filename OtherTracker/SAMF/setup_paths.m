function setup_paths()
%SETUP_PATHS Prepare SAMF paths and MEX/DLL resolution for local benchmarking.

root_path = fileparts(mfilename('fullpath'));
addpath(root_path);
addpath(fullfile(root_path, 'utility'));
addpath(fullfile(root_path, 'data'));

current_path = getenv('PATH');
path_entries = {
    root_path
    fullfile(root_path, 'utility')
    fullfile(root_path, 'data')
    };

for i = 1:numel(path_entries)
    entry = path_entries{i};
    if exist(entry, 'dir') && isempty(strfind(lower(current_path), lower(entry))) %#ok<STREMP>
        current_path = [entry pathsep current_path];
    end
end

setenv('PATH', current_path);
end
