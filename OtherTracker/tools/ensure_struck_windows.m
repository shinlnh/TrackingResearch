function ensure_struck_windows(struck_root)
%ENSURE_STRUCK_WINDOWS Build a Windows Struck executable when missing.

if nargin < 1 || exist(struck_root, 'dir') ~= 7
    error('ensure_struck_windows:missingRoot', 'Missing Struck root: %s', struck_root);
end

prepend_path_once('C:\opencv\opencv_build\cuda\install\x64\vc17\bin');

exe_candidates = { ...
    fullfile(struck_root, 'struck.exe'), ...
    fullfile(struck_root, 'build', 'bin', 'struck.exe'), ...
    fullfile(struck_root, 'build', 'bin', 'Release', 'struck.exe'), ...
    fullfile(struck_root, 'build-win64', 'bin', 'struck.exe'), ...
    fullfile(struck_root, 'build-win64', 'bin', 'Release', 'struck.exe'), ...
    fullfile(struck_root, 'build-win64', 'Release', 'struck.exe')};

for i = 1:numel(exe_candidates)
    if exist(exe_candidates{i}, 'file') == 2
        return;
    end
end

build_dir = fullfile(struck_root, 'build-win64');
if exist(build_dir, 'dir') ~= 7
    mkdir(build_dir);
end

opencv_dir = 'C:\opencv\opencv_build\cuda\install';
eigen_dir = 'C:\Program Files\MATLAB\R2024b\toolbox\shared\robotics\externalDependency\eigen\include\eigen3';

configure_cmd = sprintf([ ...
    'cmake -S "%s" -B "%s" -G "Visual Studio 17 2022" -A x64 ' ...
    '-DOpenCV_DIR="%s" -DEIGEN_INCLUDE_DIR="%s"'], ...
    struck_root, build_dir, opencv_dir, eigen_dir);
[status, output] = system(configure_cmd);
if status ~= 0
    error('ensure_struck_windows:configureFailed', ...
        'Struck CMake configure failed:\n%s', output);
end

build_cmd = sprintf('cmake --build "%s" --config Release', build_dir);
[status, output] = system(build_cmd);
if status ~= 0
    error('ensure_struck_windows:buildFailed', ...
        'Struck CMake build failed:\n%s', output);
end

for i = 1:numel(exe_candidates)
    if exist(exe_candidates{i}, 'file') == 2
        if ~strcmpi(exe_candidates{i}, exe_candidates{1})
            copyfile(exe_candidates{i}, exe_candidates{1});
        end
        return;
    end
end

error('ensure_struck_windows:missingExe', ...
    'Struck build finished but no struck.exe was found.');
end

function prepend_path_once(path_to_add)
current_path = getenv('PATH');
if contains(lower(current_path), lower(path_to_add))
    return;
end
setenv('PATH', [path_to_add pathsep current_path]);
end
