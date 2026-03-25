
function initial_net(pathModel)


global net;
global hdt_use_gpu;

net = load(pathModel);
net.layers(37+1:end)=[];
net = normalize_legacy_model(net);
hdt_use_gpu = false;
prefer_gpu = strcmp(getenv('HDT_USE_GPU'), '1');

if prefer_gpu
    try
        parallel.gpu.enableCUDAForwardCompatibility(true);
        gpuDevice();
        net = vl_simplenn_move(net, 'gpu');
        hdt_use_gpu = true;
    catch err
        warning('HDT:gpuFallback', 'Falling back to CPU mode: %s', err.message);
        net = vl_simplenn_move(net, 'cpu');
    end
else
    net = vl_simplenn_move(net, 'cpu');
end

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
