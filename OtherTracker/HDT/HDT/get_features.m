function feat = get_features(im, cos_window, layers)
%GET_FEATURES
%   Extracts dense features from image.


global net
global hdt_use_gpu

sz_window=size(cos_window);

img = single(im); % note: 255 range
img = imResample(img, net.normalization.imageSize(1:2));
img = img - net.normalization.averageImage;
if hdt_use_gpu
    img = gpuArray(img);
end

% run the CNN
res=vl_simplenn(net,img);

feat={};

for ii=1:length(layers)
    
    x = res(layers(ii)).x;
    if hdt_use_gpu
        x = gather(x);
    end
    
    x = imResample(x, sz_window(1:2));
    
    
    %process with cosine window if needed
    if ~isempty(cos_window),
        x = bsxfun(@times, x, cos_window);
    end
    
    feat{ii}=x;
    
end
end
