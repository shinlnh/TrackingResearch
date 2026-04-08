function [X, Y, Width, Height] = fDSST(frame, video, new_x, new_y, new_w, new_h)
global gconfig;

% fDSST tracker, M. Danelljan et al., TPAMI, 2017
% revised by Heng Fan, 2017

if frame == 1   % for the first frame, finish all initialization
    gconfig.base_path = 'E:\TrackingBenchmark13\';  % path to you videos
    gconfig.params.num_compressed_dim = 18;
    gconfig.translation_model_max_area = inf;
    gconfig.params.scale_model_factor = 1.0;
    
    gconfig.video = video;
    
    gconfig.video_path = [gconfig.base_path gconfig.video '\'];
    [img_files, pos, target_sz, gconfig.video_path] = load_video_info(gconfig.video_path);
    
    gconfig.params.init_pos   = floor(pos);
    gconfig.params.wsize      = floor(target_sz);
    gconfig.params.s_frames   = img_files;
    
    gconfig.pos{1}        = floor(gconfig.params.init_pos);
    gconfig.target_sz{1}  = floor(gconfig.params.wsize * 1);
    gconfig.num_frames    = numel(gconfig.params.s_frames);
    gconfig.init_target_sz     = gconfig.target_sz{1};
    gconfig.currentScaleFactor{1} = 1.0;
    
    gconfig.base_target_sz = gconfig.target_sz{1} / gconfig.currentScaleFactor{1}; 
    gconfig.sz = floor(gconfig.base_target_sz*3);
    gconfig.output_sigma = sqrt(prod(floor(gconfig.base_target_sz/4)))/16; 
    gconfig.use_sz = floor(gconfig.sz/4);
    
    gconfig.rg = circshift(-floor((gconfig.use_sz(1)-1)/2):ceil((gconfig.use_sz(1)-1)/2), [0 -floor((gconfig.use_sz(1)-1)/2)]);
    gconfig.cg = circshift(-floor((gconfig.use_sz(2)-1)/2):ceil((gconfig.use_sz(2)-1)/2), [0 -floor((gconfig.use_sz(2)-1)/2)]);
    [gconfig.rs, gconfig.cs] = ndgrid(gconfig.rg, gconfig.cg); 
    gconfig.y = exp(-0.5*(((gconfig.rs.^2 + gconfig.cs.^2)/gconfig.output_sigma^2)));
    gconfig.yf = single(fft2(gconfig.y)); 
    gconfig.interp_sz = size(gconfig.y)*4;
    
    gconfig.cos_window = single(hann(floor(gconfig.sz(1)/4))*hann(floor(gconfig.sz(2)/4))' );
    gconfig.scale_sigma = 33/16; gconfig.scale_exp = (-8:8) * 33/17;
    gconfig.scale_exp_shift = circshift(gconfig.scale_exp, [0 -8]); 
    gconfig.interp_scale_exp = -16:16;
    gconfig.interp_scale_exp_shift = circshift(gconfig.interp_scale_exp, [0 -16]); 
    gconfig.scaleSizeFactors = 1.02 .^ gconfig.scale_exp;
    
    gconfig.interpScaleFactors = 1.02 .^ gconfig.interp_scale_exp_shift; 
    gconfig.ys = exp(-0.5 * (gconfig.scale_exp_shift.^2)/gconfig.scale_sigma^2); 
    gconfig.ysf = single(fft(gconfig.ys)); 
    gconfig.scale_window = single(hann(size(gconfig.ysf,2)))';
    
    if gconfig.params.scale_model_factor^2 * prod(gconfig.init_target_sz) > 512
        gconfig.params.scale_model_factor = sqrt(512/prod(gconfig.init_target_sz));
    end
    
    gconfig.scale_model_sz = floor(gconfig.init_target_sz * gconfig.params.scale_model_factor); 
    im = imread([gconfig.video_path gconfig.params.s_frames{1}]);
    
    gconfig.min_scale_factor = 1.02 ^ ceil(log(max(5 ./ gconfig.sz)) / log(1.02));
    gconfig.max_scale_factor = 1.02 ^ floor(log(min([size(im,1) size(im,2)] ./ gconfig.base_target_sz)) / log(1.02));
    
    gconfig.projection_matrix = [];
    gconfig.rect_position = zeros(gconfig.num_frames, 4);
end

result = track(frame, new_x, new_y, new_w, new_h);
X = result(1); Y = result(2); Width = result(3); Height = result(4);
end

function result = track(frame, new_x, new_y, new_w, new_h)
global gconfig;
im = imread([gconfig.video_path gconfig.params.s_frames{frame}]);
gconfig.im{frame} = im;

%do tracking
flag = 0;
if (new_x == 0 && new_y == 0 && new_w ==0 && new_h == 0)
    flag = 1;
end
if frame > 1
    if flag == 1
        %translation search
        [xt_npca, xt_pca] = get_subwindow(gconfig.im{frame}, gconfig.pos{frame-1}, gconfig.sz, gconfig.currentScaleFactor{frame-1});
        xt = feature_projection(xt_npca, xt_pca, gconfig.projection_matrix, gconfig.cos_window);
        xtf = fft2(xt);   
        responsef = resizeDFT2(sum(gconfig.hf_num{frame-1} .* xtf, 3) ./ (gconfig.hf_den{frame-1} + 0.01), gconfig.interp_sz);
        response = ifft2(responsef, 'symmetric');
            
        [row, col] = find(response == max(response(:)), 1);
        disp_row = mod(row - 1 + floor((gconfig.interp_sz(1)-1)/2), gconfig.interp_sz(1)) - floor((gconfig.interp_sz(1)-1)/2);
        disp_col = mod(col - 1 + floor((gconfig.interp_sz(2)-1)/2), gconfig.interp_sz(2)) - floor((gconfig.interp_sz(2)-1)/2);
            
        translation_vec = round([disp_row, disp_col] * gconfig.currentScaleFactor{frame-1});
        gconfig.pos{frame} = gconfig.pos{frame-1} + translation_vec;   % new position of object
    else
        % detection result: [new_x, new_y, new_w, new_h]
        % we need to reset the position of object based on detection result
        gconfig.pos{frame} = [new_y+new_h/2 new_x+new_w/2];
%         disp('Just updating!');
%         disp(gconfig.pos{frame});
    end
    %scale search
    [xs_pca, xs_npca] = get_scale_subwindow(gconfig.im{frame}, gconfig.pos{frame}, gconfig.base_target_sz,...
                                            gconfig.currentScaleFactor{frame-1}*gconfig.scaleSizeFactors,gconfig.scale_model_sz);

    xs  = feature_projection_scale(xs_npca, xs_pca, gconfig.scale_basis{frame-1}, gconfig.scale_window);
    xsf = fft(xs,[], 2);
            
    scale_responsef = sum(gconfig.sf_num{frame-1} .* xsf, 1) ./ (gconfig.sf_den{frame-1} + 0.01);
            
    interp_scale_response = ifft(resizeDFT(scale_responsef, 33), 'symmetric');
            
    recovered_scale_index = find(interp_scale_response == max(interp_scale_response(:)), 1);
        
    %set the scale
    gconfig.currentScaleFactor{frame} = gconfig.currentScaleFactor{frame-1} * gconfig.interpScaleFactors(recovered_scale_index);
    
    %adjust to make sure we are not to large or to small
    if gconfig.currentScaleFactor{frame} < gconfig.min_scale_factor
        gconfig.currentScaleFactor{frame} = gconfig.min_scale_factor;
    elseif gconfig.currentScaleFactor{frame} > gconfig.max_scale_factor
        gconfig.currentScaleFactor{frame} = gconfig.max_scale_factor;
    end
end
    
% compute coefficients for the tranlsation filter
[xl_npca, xl_pca] = get_subwindow(gconfig.im{frame}, gconfig.pos{frame}, gconfig.sz, gconfig.currentScaleFactor{frame});

if frame == 1
    gconfig.h_num_pca{frame}  = xl_pca;
    gconfig.h_num_npca{frame} = xl_npca;
        
    % set number of compressed dimensions to maximum if too many
    gconfig.params.num_compressed_dim = min(gconfig.params.num_compressed_dim, size(xl_pca, 2));
else
    gconfig.h_num_pca{frame}  = (1 - 0.025) * gconfig.h_num_pca{frame-1} + 0.025 * xl_pca;
    gconfig.h_num_npca{frame} = (1 - 0.025) * gconfig.h_num_npca{frame-1} + 0.025 * xl_npca;
end
    
data_matrix = gconfig.h_num_pca{frame};
    
[pca_basis, ~, ~] = svd(data_matrix' * data_matrix);
gconfig.projection_matrix = pca_basis(:, 1:gconfig.params.num_compressed_dim);
    
hf_proj = fft2(feature_projection(gconfig.h_num_npca{frame}, gconfig.h_num_pca{frame}, gconfig.projection_matrix, gconfig.cos_window));
gconfig.hf_num{frame} = bsxfun(@times, gconfig.yf, conj(hf_proj));
    
xlf = fft2(feature_projection(xl_npca, xl_pca, gconfig.projection_matrix, gconfig.cos_window));
new_hf_den = sum(xlf .* conj(xlf), 3);
if frame == 1
    gconfig.hf_den{frame} = new_hf_den;
else
    gconfig.hf_den{frame} = (1 - 0.025) * gconfig.hf_den{frame-1} + 0.025 * new_hf_den;
end
    
% compute coefficents for the scale filter
[xs_pca, xs_npca] = get_scale_subwindow(gconfig.im{frame}, gconfig.pos{frame}, gconfig.base_target_sz, ...
                                        gconfig.currentScaleFactor{frame}*gconfig.scaleSizeFactors, gconfig.scale_model_sz);
if frame == 1
    gconfig.s_num{frame} = xs_pca;
else
    gconfig.s_num{frame} = (1 - 0.025) * gconfig.s_num{frame-1} + 0.025 * xs_pca;
end;
        
bigY = gconfig.s_num{frame};
bigY_den = xs_pca;

[gconfig.scale_basis{frame}, ~] = qr(bigY, 0);
[scale_basis_den, ~] = qr(bigY_den, 0);
gconfig.scale_basis{frame} = gconfig.scale_basis{frame}';
        
%create the filter update coefficients
sf_proj = fft(feature_projection_scale([], gconfig.s_num{frame}, gconfig.scale_basis{frame}, gconfig.scale_window),[],2);
gconfig.sf_num{frame} = bsxfun(@times, gconfig.ysf, conj(sf_proj));
        
xs = feature_projection_scale(xs_npca, xs_pca, scale_basis_den', gconfig.scale_window);
xsf = fft(xs,[],2);
new_sf_den = sum(xsf .* conj(xsf),1);
if frame == 1
    gconfig.sf_den{frame} = new_sf_den;
else
    gconfig.sf_den{frame} = (1 - 0.025) * gconfig.sf_den{frame-1} + 0.025 * new_sf_den;
end
    
gconfig.target_sz{frame} = floor(gconfig.base_target_sz * gconfig.currentScaleFactor{frame});
% disp(gconfig.target_sz{frame});
% result = [pos(1) pos(2) target_sz(1) target_sz(2)]
% result = [gconfig.pos{frame}([2,1]) - floor(gconfig.target_sz{frame}([2,1])/2), gconfig.target_sz{frame}([2,1])];
result = [gconfig.pos{frame}, gconfig.target_sz{frame}];
% disp(result);
end

function [out_pca,out_npca] = get_scale_subwindow(im, pos, base_target_sz, scaleFactors, scale_model_sz)
nScales = length(scaleFactors);
for s = 1:nScales
    patch_sz = floor(base_target_sz * scaleFactors(s));
    
    xs = floor(pos(2)) + (1:patch_sz(2)) - floor(patch_sz(2)/2);
    ys = floor(pos(1)) + (1:patch_sz(1)) - floor(patch_sz(1)/2);
   
    xs(xs < 1) = 1;
    ys(ys < 1) = 1;
    xs(xs > size(im,2)) = size(im,2);
    ys(ys > size(im,1)) = size(im,1);
    
    %extract image
    im_patch = im(ys, xs, :);
    
    % resize image to model size
    im_patch_resized = mexResize(im_patch, scale_model_sz, 'auto');
    
    % extract scale features
    temp_hog = fhog(single(im_patch_resized), 4);
    
    if s == 1
        dim_scale = size(temp_hog,1)*size(temp_hog,2)*31;
        out_pca = zeros(dim_scale, nScales, 'single');
    end
    out_pca(:,s) = reshape(temp_hog(:,:,1:31), dim_scale, 1);
end
out_npca = [];
end

function [out_npca, out_pca] = get_subwindow(im, pos, model_sz, currentScaleFactor)
if isscalar(model_sz)
    model_sz = [model_sz, model_sz];
end
patch_sz = floor(model_sz * currentScaleFactor);
if patch_sz(1) < 1
    patch_sz(1) = 2;
end;
if patch_sz(2) < 1
    patch_sz(2) = 2;
end;
xs = floor(pos(2)) + (1:patch_sz(2)) - floor(patch_sz(2)/2);
ys = floor(pos(1)) + (1:patch_sz(1)) - floor(patch_sz(1)/2);
xs(xs < 1) = 1;
ys(ys < 1) = 1;
xs(xs > size(im,2)) = size(im,2);
ys(ys > size(im,1)) = size(im,1);
%extract image
im_patch = im(ys, xs, :);
%resize image to model size
im_patch = mexResize(im_patch, model_sz, 'auto');
% compute non-pca feature map
out_npca = [];
% compute pca feature map
temp_pca = fhog(single(im_patch),4, 9);
temp_pca(:,:,32) = cell_grayscale(im_patch,4);
out_pca = reshape(temp_pca, [size(temp_pca, 1)*size(temp_pca, 2), size(temp_pca, 3)]);
end

function H = fhog( I, binSize, nOrients, clip, crop )
if( nargin<2 ), binSize=8; end
if( nargin<3 ), nOrients=9; end
if( nargin<4 ), clip=.2; end
if( nargin<5 ), crop=0; end
softBin = -1; useHog = 2; b = binSize;
[M,O]=gradientMex('gradientMag',I,0,1);
H = gradientMex('gradientHist',M,O,binSize,nOrients,softBin,useHog,clip);
if( crop ), e=mod(size(I),b)<b/2; H=H(2:end-e(1),2:end-e(2),:); end
end

function [ cell_gray ] = cell_grayscale( img, w )
if size(img,3) == 3
   %convert to grayscale
   gray_image = rgb2gray(img);
else
   gray_image = img;
end
gray_image = single(gray_image);
iImage = integralImage(gray_image);
i1 = (w:w:size(gray_image,1)) + 1;
i2 = (w:w:size(gray_image,2)) + 1;
cell_sum = iImage(i1,i2) - iImage(i1,i2-w) - iImage(i1-w,i2) + iImage(i1-w,i2-w);
cell_gray = cell_sum / (w*w * 255) - 0.5;
end

function z = feature_projection(x_npca, x_pca, projection_matrix, cos_window)
[height, width] = size(cos_window);
[num_pca_in, num_pca_out] = size(projection_matrix);
z = bsxfun(@times, cos_window, reshape(x_pca * projection_matrix, [height, width, num_pca_out]));
end

function z = feature_projection_scale(x_npca, x_pca, projection_matrix, cos_window)
z = bsxfun(@times, cos_window, projection_matrix * x_pca);
end

function [img_files, pos, target_sz, video_path] = load_video_info(video)
	%full path to the video's files
	video_path = video;

	filename = [video_path 'groundtruth_rect.txt'];
    ground_truth = dlmread(filename);
	
	target_sz = [ground_truth(1,4), ground_truth(1,3)];
	pos = [ground_truth(1,2), ground_truth(1,1)] + floor(target_sz/2);
	
	%from now on, work in the subfolder where all the images are
	video_path = [video_path 'img/'];
	
    video_name = regexp(video, '\', 'split');
    video_name = video_name{3};
	frames = {'Football1', 1, 74; 'Freeman3', 1, 460; 'Freeman4', 1, 283};
	
	idx = find(strcmpi(video_name, frames(:,1)));

    if isempty(idx)
		%general case, just list all images
		img_files = dir([video_path '*.png']);
		if isempty(img_files),
			img_files = dir([video_path '*.jpg']);
			assert(~isempty(img_files), 'No image files to load.')
		end
		img_files = sort({img_files.name});
    else
        if exist(sprintf('%s%04i.png', video_path, frames{idx,2}), 'file'),
			img_files = num2str((frames{idx,2} : frames{idx,3})', '%04i.png');
			
		elseif exist(sprintf('%s%04i.jpg', video_path, frames{idx,2}), 'file'),
			img_files = num2str((frames{idx,2} : frames{idx,3})', '%04i.jpg');
		else
			error('No image files to load.')
        end
		img_files = cellstr(img_files);
    end
end

function resizeddft  = resizeDFT(inputdft, desiredLen)
len = length(inputdft);
minsz = min(len, desiredLen);
scaling = desiredLen/len;
if size(inputdft, 1) > 1
    newSize = [desiredLen 1];
else
    newSize = [1 desiredLen];
end
resizeddft = complex(zeros(newSize, 'single'));
mids = ceil(minsz/2);
mide = floor((minsz-1)/2) - 1;
resizeddft(1:mids) = scaling * inputdft(1:mids);
resizeddft(end - mide:end) = scaling * inputdft(end - mide:end);
end

function resizeddft  = resizeDFT2(inputdft, desiredSize)
imsz = size(inputdft);
minsz = min(imsz, desiredSize);
scaling = prod(desiredSize)/prod(imsz);
resizeddft = complex(zeros(desiredSize, 'single'));
mids = ceil(minsz/2);
mide = floor((minsz-1)/2) - 1;
resizeddft(1:mids(1), 1:mids(2)) = scaling * inputdft(1:mids(1), 1:mids(2));
resizeddft(1:mids(1), end - mide(2):end) = scaling * inputdft(1:mids(1), end - mide(2):end);
resizeddft(end - mide(1):end, 1:mids(2)) = scaling * inputdft(end - mide(1):end, 1:mids(2));
resizeddft(end - mide(1):end, end - mide(2):end) = scaling * inputdft(end - mide(1):end, end - mide(2):end);
end