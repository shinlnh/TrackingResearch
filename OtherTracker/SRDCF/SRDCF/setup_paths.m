function setup_paths()

% Add the tracker root and expose shipped DLLs for old MEX binaries.

pathstr = fileparts(mfilename('fullpath'));

current_path = getenv('PATH');
if isempty(strfind(lower(current_path), lower(pathstr))) %#ok<STREMP>
    current_path = [pathstr pathsep current_path];
    setenv('PATH', current_path);
end

addpath(pathstr);
end
