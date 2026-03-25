function build_matconvnet_gpu(cuda_root, gpu_id)
%BUILD_MATCONVNET_GPU Compile MatConvNet with GPU support for CCOT.

if nargin < 1 || isempty(cuda_root)
    cuda_root = getenv('CUDA_PATH_V13_1');
    if isempty(cuda_root)
        cuda_root = getenv('CUDA_PATH');
    end
end

if nargin < 2 || isempty(gpu_id)
    gpu_id = 1;
end

if isempty(cuda_root)
    error('build_matconvnet_gpu:missingCudaRoot', 'CUDA root not found in the environment.');
end

parallel.gpu.enableCUDAForwardCompatibility(true);
gpu_device = gpuDevice(gpu_id);
arch_code = strrep(gpu_device.ComputeCapability, '.', '');
cuda_arch = sprintf('-gencode=arch=compute_%s,code=\\\"sm_%s,compute_%s\\\" ', ...
    arch_code, arch_code, arch_code);

cc = mex.getCompilerConfigurations('C++', 'Selected');
if isempty(cc)
    error('build_matconvnet_gpu:noCompiler', 'No selected C++ compiler for MEX.');
end

cl_candidates = dir(fullfile(cc.Location, 'VC', 'Tools', 'MSVC', '*', 'bin', 'Hostx64', 'x64', 'cl.exe'));
if isempty(cl_candidates)
    error('build_matconvnet_gpu:noCl', 'Failed to locate cl.exe under %s.', cc.Location);
end

cl_dir = cl_candidates(1).folder;
path_sep = ';';
current_path = getenv('PATH');
if isempty(regexpi(current_path, regexptranslate('escape', cl_dir), 'once'))
    setenv('PATH', [cl_dir path_sep current_path]);
end

matlab_dir = fullfile(fileparts(mfilename('fullpath')), 'external_libs', 'matconvnet', 'matlab');
old_dir = pwd;
cleanup_obj = onCleanup(@() cd(old_dir)); %#ok<NASGU>
cd(matlab_dir);

vl_compilenn('enableGpu', true, 'cudaRoot', cuda_root, 'cudaArch', cuda_arch);
movefile(fullfile('mex', 'vl_*.mex*'), '.');
end
