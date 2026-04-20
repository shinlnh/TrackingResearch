function ensure_cfnet_assets(cfnet_root)
%ENSURE_CFNET_ASSETS Verify CFNet pretrained assets and synthesize minimal stats.

if nargin < 1 || isempty(cfnet_root)
    error('ensure_cfnet_assets:rootRequired', 'cfnet_root is required');
end

net_path = fullfile(cfnet_root, 'pretrained', 'networks', 'cfnet-conv2_e80.mat');
net_gray_path = fullfile(cfnet_root, 'pretrained', 'networks', 'cfnet-conv2_gray_e40.mat');
stats_path = fullfile(cfnet_root, 'data', 'ILSVRC2015.stats.mat');

assert(exist(net_path, 'file') == 2, 'Missing CFNet pretrained model: %s', net_path);
assert(exist(net_gray_path, 'file') == 2, 'Missing CFNet gray pretrained model: %s', net_gray_path);

if exist(stats_path, 'file') ~= 2
    x = struct('rgbMean', single([0 0 0])); %#ok<NASGU>
    z = struct('rgbMean', single([0 0 0])); %#ok<NASGU>
    save(stats_path, 'x', 'z');
end
end
