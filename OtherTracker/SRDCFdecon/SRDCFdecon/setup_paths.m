function setup_paths()

% Add the neccesary paths

[pathstr, name, ext] = fileparts(mfilename('fullpath'));

% Ensure dependent DLLs for the shipped MEX files are visible to Windows.
path_entries = {
    pathstr, ...
    [pathstr '/mexResize/'], ...
    [pathstr '/feature_extraction/'], ...
    [pathstr '/external_libs/'] ...
};
current_path = getenv('PATH');
for i = 1:numel(path_entries)
    entry = path_entries{i};
    if isempty(strfind(lower(current_path), lower(entry))) %#ok<STREMP>
        current_path = [entry pathsep current_path];
    end
end
setenv('PATH', current_path);

% Tracker implementation
addpath([pathstr '/implementation/']);

% Runfiles
addpath([pathstr '/runfiles/']);

% Utilities
addpath([pathstr '/utils/']);

% The feature extraction
addpath(genpath([pathstr '/feature_extraction/']));

% Mtimesx
addpath([pathstr '/external_libs/']);

% mexResize
addpath([pathstr '/mexResize/']);
