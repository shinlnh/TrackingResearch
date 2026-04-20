function initial_net()
% INITIAL_NET: Loading VGG-Net-19

global net;
utility_dir = fileparts(mfilename('fullpath'));
tracker_root = fileparts(utility_dir);
verified_root = fileparts(tracker_root);
othertracker_root = fileparts(verified_root);

candidate_paths = { ...
    fullfile(tracker_root, 'vgg_model', 'imagenet-vgg-verydeep-19.mat'), ...
    fullfile(othertracker_root, 'HDT', 'HDT', 'imagenet-vgg-verydeep-19.mat'), ...
    'imagenet-vgg-verydeep-19.mat'};

model_path = '';
for i = 1:numel(candidate_paths)
    if exist(candidate_paths{i}, 'file') == 2
        model_path = candidate_paths{i};
        break;
    end
end

assert(~isempty(model_path), 'HCFT:missingModel', ...
    'imagenet-vgg-verydeep-19.mat not found in HCFT or shared repo locations.');
net = load(model_path);

% Remove the fully connected layers and classification layer
net.layers(37+1:end) = [];

% Switch to GPU mode
global enableGPU;
if enableGPU
    net = vl_simplenn_move(net, 'gpu');
end

net=vl_simplenn_tidy(net);

end
