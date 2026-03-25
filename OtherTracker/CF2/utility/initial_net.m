function initial_net()
% INITIAL_NET: Loading VGG-Net-19

global net;
global enableGPU;

model_path = resolve_model_path();
net = load(model_path);

% Remove the fully connected layers and classification layer
net.layers(37+1:end) = [];
net = normalize_legacy_model(net);

if enableGPU
    try
        parallel.gpu.enableCUDAForwardCompatibility(true);
        gpuDevice();
        net = vl_simplenn_move(net, 'gpu');
    catch err
        warning('CF2:gpuFallback', 'Falling back to CPU mode: %s', err.message);
        enableGPU = false;
        net = vl_simplenn_move(net, 'cpu');
    end
else
    net = vl_simplenn_move(net, 'cpu');
end

net = vl_simplenn_tidy(net);

end

function model_path = resolve_model_path()
local_model = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'model', 'imagenet-vgg-verydeep-19.mat');
if exist(local_model, 'file')
    model_path = local_model;
    return;
end

fallback_model = fullfile(fileparts(fileparts(fileparts(mfilename('fullpath')))), 'HDT', 'HDT', 'imagenet-vgg-verydeep-19.mat');
if exist(fallback_model, 'file')
    model_path = fallback_model;
    return;
end

error('CF2:modelMissing', ['Missing imagenet-vgg-verydeep-19.mat. ' ...
    'Place it in OtherTracker/CF2/model or reuse the HDT copy.']);
end

function net = normalize_legacy_model(net)
for i = 1:numel(net.layers)
    layer = net.layers{i};
    if isfield(layer, 'weights') && ~isfield(layer, 'filters') && numel(layer.weights) >= 2
        layer.filters = layer.weights{1};
        layer.biases = layer.weights{2};
    end
    net.layers{i} = layer;
end
end
