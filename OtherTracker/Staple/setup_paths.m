function setup_paths()

% Expose the tracker root so shipped DLLs and MEX files can be resolved.

pathstr = fileparts(mfilename('fullpath'));

current_path = getenv('PATH');
if isempty(strfind(lower(current_path), lower(pathstr))) %#ok<STREMP>
    current_path = [pathstr pathsep current_path];
    setenv('PATH', current_path);
end

addpath(pathstr);
end
