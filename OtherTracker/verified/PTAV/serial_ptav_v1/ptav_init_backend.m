function ptav_init_backend()
%PTAV_INIT_BACKEND Initialize the Python OpenCV DNN backend used instead of MatCaffe.

global ptav_py_backend;

repo_root = fileparts(fileparts(fileparts(fileparts(fileparts(mfilename('fullpath'))))));
python_exe = fullfile(repo_root, 'OtherTracker', 'verified', 'StructSiam', '.venv39', 'Scripts', 'python.exe');

pe = pyenv;
if pe.Status == "NotLoaded"
    pyenv('Version', python_exe);
elseif isprop(pe, 'Executable') && pe.Executable ~= string(python_exe)
    error('ptav_init_backend:pythonMismatch', ...
        'MATLAB already loaded a different Python executable: %s', pe.Executable);
end

backend_dir = string(fileparts(mfilename('fullpath')));
sys = py.importlib.import_module('sys');
sys.path.insert(int32(0), char(backend_dir));

ptav_py_backend = py.importlib.import_module('ptav_dnn_backend');
py.importlib.reload(ptav_py_backend);
ptav_py_backend.init_backend();
try
    runtime_desc = string(ptav_py_backend.get_runtime_desc());
    fprintf('[ptav] verifier_runtime=%s\n', runtime_desc);
catch
end
end
