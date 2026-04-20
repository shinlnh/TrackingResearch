function ensure_traca_assets(traca_root)
%ENSURE_TRACA_ASSETS Populate missing TRACA network assets from the local download cache.

if nargin < 1 || isempty(traca_root)
    error('ensure_traca_assets:rootRequired', 'traca_root is required');
end

network_dir = fullfile(traca_root, 'network');
required = { ...
    fullfile(network_dir, 'imagenet-vgg-m-2048.mat'), ...
    fullfile(network_dir, 'multi_daenet.mat'), ...
    fullfile(network_dir, 'prior_network.mat')};

if all(cellfun(@(p) exist(p, 'file') == 2, required))
    return;
end

repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
cached_network_dir = fullfile(repo_root, '_tmp_traca_download', 'TRACA_extracted', 'network');
if exist(cached_network_dir, 'dir') ~= 7
    error('ensure_traca_assets:missingCache', ...
        'Missing local TRACA asset cache: %s', cached_network_dir);
end

if exist(network_dir, 'dir') ~= 7
    mkdir(network_dir);
end

for i = 1:numel(required)
    [~, name, ext] = fileparts(required{i});
    source = fullfile(cached_network_dir, [name ext]);
    if exist(source, 'file') ~= 2
        error('ensure_traca_assets:missingSource', 'Missing cached TRACA asset: %s', source);
    end
    if exist(required{i}, 'file') ~= 2
        copyfile(source, required{i});
    end
end
end
