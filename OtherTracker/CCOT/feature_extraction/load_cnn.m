function net = load_cnn(fparams, im_size)

net = load(['networks/' fparams.nn_name]);
net = vl_simplenn_tidy(net);
net.layers = net.layers(1:max(fparams.output_layer));

if strcmpi(fparams.input_size_mode, 'cnn_default')
    base_input_sz = net.meta.normalization.imageSize(1:2);
elseif strcmpi(fparams.input_size_mode, 'adaptive')
    base_input_sz = im_size(1:2);
else
    error('Unknown input_size_mode');
end

net.meta.normalization.imageSize(1:2) = round(base_input_sz .* fparams.input_size_scale);
net.meta.normalization.averageImage = imresize(single(net.meta.normalization.averageImage), net.meta.normalization.imageSize(1:2));

net.info = vl_simplenn_display(net);

if isfield(fparams, 'use_gpu') && fparams.use_gpu
    persistent initialized_gpu_id;
    if isfield(fparams, 'gpu_id') && ~isempty(fparams.gpu_id)
        requested_gpu_id = fparams.gpu_id;
    else
        requested_gpu_id = 1;
    end
    parallel.gpu.enableCUDAForwardCompatibility(true);
    if isempty(initialized_gpu_id) || initialized_gpu_id ~= requested_gpu_id
        gpuDevice(requested_gpu_id);
        initialized_gpu_id = requested_gpu_id;
    end
    net = vl_simplenn_move(net, 'gpu');
end
end
