function ensure_tracker_matconvnet(matconvnet_root, use_gpu, required)
%ENSURE_TRACKER_MATCONVNET Build bundled MatConvNet for the current MATLAB arch.

if nargin < 2
    use_gpu = true;
end
if nargin < 3 || isempty(required)
    required = {'vl_nnconv', 'vl_nnconvt', 'vl_nnpool', 'vl_nnnormalize', 'vl_nnbnorm'};
end

mex_dir = fullfile(matconvnet_root, 'matlab', 'mex');
if has_required_mex(mex_dir, required)
    return;
end

if exist(matconvnet_root, 'dir') ~= 7
    error('ensure_tracker_matconvnet:missingRoot', 'Missing MatConvNet root: %s', matconvnet_root);
end

old_dir = pwd;
cleanup_obj = onCleanup(@() cd(old_dir)); %#ok<NASGU>
cd(matconvnet_root);
addpath(fullfile(matconvnet_root, 'matlab'));

if use_gpu
    cuda_root = detect_cuda_root();
    cuda_arch = '-gencode=arch=compute_90,code=\"sm_90,compute_90\"';
    fprintf('Building MatConvNet with GPU support in %s\n', matconvnet_root);
    vl_compilenn('EnableGpu', true, ...
        'EnableCudnn', false, ...
        'CudaRoot', cuda_root, ...
        'CudaArch', cuda_arch, ...
        'Verbose', 1);
else
    fprintf('Building MatConvNet CPU-only in %s\n', matconvnet_root);
    vl_compilenn('EnableGpu', false, 'Verbose', 1);
end

if ~has_required_mex(mex_dir, required)
    error('ensure_tracker_matconvnet:buildFailed', ...
        'MatConvNet build did not produce required mex files in %s', mex_dir);
end
end

function ok = has_required_mex(mex_dir, required)
ok = true;
for i = 1:numel(required)
    if exist(fullfile(mex_dir, [required{i} '.' mexext]), 'file') == 0
        ok = false;
        return;
    end
end
end

function cuda_root = detect_cuda_root()
candidates = { ...
    getenv('CUDA_PATH'), ...
    'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4', ...
    'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1'};

cuda_root = '';
for i = 1:numel(candidates)
    candidate = candidates{i};
    if ~isempty(candidate) && exist(fullfile(candidate, 'bin', 'nvcc.exe'), 'file') == 2
        cuda_root = candidate;
        return;
    end
end

error('ensure_tracker_matconvnet:missingCuda', ...
    'Could not find a usable CUDA toolkit with nvcc.exe.');
end
