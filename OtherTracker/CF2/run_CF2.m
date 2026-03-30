function results = run_CF2(seq, res_path, bSaveImage)

% RUN_CF2:
% process a sequence using CF2 (Correlation filter tracking with convolutional features)
%
% Input:
%     - seq:        sequence name
%     - res_path:   result path
%     - bSaveImage: flag for saving images
% Output:
%     - results: tracking results, position prediction over time
%
%   It is provided for educational/researrch purpose only.
%   If you find the software useful, please consider cite our paper.
%
%   Hierarchical Convolutional Features for Visual Tracking
%   Chao Ma, Jia-Bin Huang, Xiaokang Yang, and Ming-Hsuan Yang
%   IEEE International Conference on Computer Vision, ICCV 2015
%
% Contact:
%   Chao Ma (chaoma99@gmail.com), or
%   Jia-Bin Huang (jbhuang1@illinois.edu).

% ================================================================================
% Environment setting
% ================================================================================
global enableGPU;
global cf2_sequence_name;
global net;
persistent cf2_gpu_unavailable;

if isempty(cf2_gpu_unavailable)
    cf2_gpu_unavailable = false;
end

enableGPU = strcmp(getenv('CF2_USE_GPU'), '1') && ~cf2_gpu_unavailable;
cf2_sequence_name = seq.name;

% Image file names
img_files = seq.s_frames;
% Seletected target size
target_sz = [seq.init_rect(1,4), seq.init_rect(1,3)];
% Initial target position
pos       = [seq.init_rect(1,2), seq.init_rect(1,1)] + floor(target_sz/2);

% Extra area surrounding the target for including contexts
padding = struct('generic', 1.8, 'large', 1, 'height', 0.4);

lambda = 1e-4;              % Regularization
output_sigma_factor = 0.1;  % Spatial bandwidth (proportional to target)

interp_factor = 0.01;       % Model learning rate
cell_size = 4;              % Spatial cell size

video_path='';

show_visualization=false;

% ================================================================================
% Main entry function for visual tracking
% ================================================================================
try
    [rects, time] = tracker_ensemble(video_path, img_files, pos, target_sz, ...
        padding, lambda, output_sigma_factor, interp_factor, ...
        cell_size, show_visualization);
catch err
    if enableGPU && is_gpu_runtime_error(err)
        warning('CF2:gpuFallbackRuntime', ...
            'GPU path failed for %s. Retrying on CPU. Root cause: %s', ...
            seq.name, err.message);
        cf2_gpu_unavailable = true;
        enableGPU = false;
        setenv('CF2_USE_GPU', '0');
        net = [];
        [rects, time] = tracker_ensemble(video_path, img_files, pos, target_sz, ...
            padding, lambda, output_sigma_factor, interp_factor, ...
            cell_size, show_visualization);
    else
        rethrow(err);
    end
end

% ================================================================================
% Return results to benchmark, in a workspace variable
% ================================================================================
results.type   = 'rect';
results.res    = rects;
results.fps    = numel(img_files)/time;

cf2_sequence_name = '';

end

function tf = is_gpu_runtime_error(err)
report = lower(getReport(err, 'extended', 'hyperlinks', 'off'));
patterns = {
    'compute capability'
    'cuda'
    'gpu device'
    'parallel.gpu.enablecudaforwardcompatibility'
    'vl_nnconv'
};
tf = false;
for i = 1:numel(patterns)
    if contains(report, patterns{i})
        tf = true;
        return;
    end
end
end

