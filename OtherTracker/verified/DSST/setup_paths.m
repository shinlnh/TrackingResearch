function setup_paths()
%SETUP_PATHS Add DSST code and its DLL folder for local MATLAB runs.

root_path = fileparts(mfilename('fullpath'));
code_path = fullfile(root_path, 'code');

addpath(root_path);
addpath(code_path);

current_path = getenv('PATH');
if isempty(strfind(lower(current_path), lower(code_path))) %#ok<STREMP>
    setenv('PATH', [code_path pathsep current_path]);
end
end
