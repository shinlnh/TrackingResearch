function res = vl_simplenn_compat(net, x, dzdy, res, varargin)
%VL_SIMPLENN_COMPAT Compatibility wrapper for older VITAL callsites.

for layer_idx = 1:numel(net.layers)
    layer = net.layers{layer_idx};
    if isfield(layer, 'filters')
        if isfield(layer, 'biases')
            net.layers{layer_idx}.weights = {layer.filters, layer.biases};
        else
            net.layers{layer_idx}.weights = {layer.filters, []};
        end
    end
    if isfield(layer, 'filtersLearningRate')
        bias_lr = 0;
        if isfield(layer, 'biasesLearningRate')
            bias_lr = layer.biasesLearningRate;
        end
        net.layers{layer_idx}.learningRate = [layer.filtersLearningRate, bias_lr];
    end
    if isfield(layer, 'filtersWeightDecay')
        bias_wd = 0;
        if isfield(layer, 'biasesWeightDecay')
            bias_wd = layer.biasesWeightDecay;
        end
        net.layers{layer_idx}.weightDecay = [layer.filtersWeightDecay, bias_wd];
    end
end

filtered = {};
i = 1;
while i <= numel(varargin)
    arg = varargin{i};
    if (ischar(arg) || isstring(arg)) && i < numel(varargin)
        key = char(arg);
        if any(strcmpi(key, {'disableDropout', 'freezeDropout'}))
            i = i + 2;
            continue;
        end
    end
    filtered{end + 1} = varargin{i}; %#ok<AGROW>
    i = i + 1;
end

res = vl_simplenn(net, x, dzdy, res, filtered{:});
end
