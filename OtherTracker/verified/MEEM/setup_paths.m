function setup_paths()
%SETUP_PATHS Add the local MEEM mirror and its subdirectories.

root_path = fileparts(mfilename('fullpath'));
upstream_path = fullfile(root_path, 'upstream');

addpath(root_path);
addpath(genpath(upstream_path));
end
