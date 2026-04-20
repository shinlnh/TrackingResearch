function results = PTAV(params, verify_param)
global net;
% global sim_score;

s_frames      = params.s_frames;
pos           = floor(params.init_pos);
target_sz     = floor(params.wsize * params.resize_factor);

visualization = params.visualization;

num_frames = numel(s_frames);

init_target_sz = target_sz;

if prod(init_target_sz) > params.translation_model_max_area
    currentScaleFactor = sqrt(prod(init_target_sz) / params.translation_model_max_area);
else
    currentScaleFactor = 1.0;
end

% target size at the initial scale
base_target_sz = target_sz / currentScaleFactor;

%window size, taking padding into account
sz = floor( base_target_sz * (1 + params.padding ));

featureRatio = 4;

output_sigma = sqrt(prod(floor(base_target_sz/featureRatio))) * params.output_sigma_factor;
use_sz = floor(sz/featureRatio);
rg = circshift(-floor((use_sz(1)-1)/2):ceil((use_sz(1)-1)/2), [0 -floor((use_sz(1)-1)/2)]);
cg = circshift(-floor((use_sz(2)-1)/2):ceil((use_sz(2)-1)/2), [0 -floor((use_sz(2)-1)/2)]);

[rs, cs] = ndgrid( rg,cg);
y = exp(-0.5 * (((rs.^2 + cs.^2) / output_sigma^2)));
yf = single(fft2(y));

interp_sz = size(y) * featureRatio;

cos_window = single(hann(floor(sz(1)/featureRatio))*hann(floor(sz(2)/featureRatio))' );

if params.number_of_scales > 0
    scale_sigma = params.number_of_interp_scales * params.scale_sigma_factor;
    
    scale_exp = (-floor((params.number_of_scales-1)/2):ceil((params.number_of_scales-1)/2)) * params.number_of_interp_scales/params.number_of_scales;
    scale_exp_shift = circshift(scale_exp, [0 -floor((params.number_of_scales-1)/2)]);
    
    interp_scale_exp = -floor((params.number_of_interp_scales-1)/2):ceil((params.number_of_interp_scales-1)/2);
    interp_scale_exp_shift = circshift(interp_scale_exp, [0 -floor((params.number_of_interp_scales-1)/2)]);
    
    scaleSizeFactors = params.scale_step .^ scale_exp;
    interpScaleFactors = params.scale_step .^ interp_scale_exp_shift;
    
    ys = exp(-0.5 * (scale_exp_shift.^2) /scale_sigma^2);
    ysf = single(fft(ys));
    scale_window = single(hann(size(ysf,2)))';
    
    %make sure the scale model is not to large, to save computation time
    if params.scale_model_factor^2 * prod(init_target_sz) > params.scale_model_max_area
        params.scale_model_factor = sqrt(params.scale_model_max_area/prod(init_target_sz));
    end
    
    %set the scale model size
    scale_model_sz = floor(init_target_sz * params.scale_model_factor);
    
    im = imread([params.video_path s_frames{1}]);
    
    %force reasonable scale changes
    min_scale_factor = params.scale_step ^ ceil(log(max(5 ./ sz)) / log(params.scale_step));
    max_scale_factor = params.scale_step ^ floor(log(min([size(im,1) size(im,2)] ./ base_target_sz)) / log(params.scale_step));
    
    max_scale_dim = strcmp(params.s_num_compressed_dim,'MAX');
    if max_scale_dim
        s_num_compressed_dim = length(scaleSizeFactors);
    else
        s_num_compressed_dim = params.s_num_compressed_dim;
    end
end

% initialize the projection matrix
projection_matrix = [];

rect_position = zeros(num_frames, 4);

for frame = 1:num_frames,
    %load image
    im = imread([params.video_path s_frames{frame}]);
    
    verification_result = true;
    
    %do tracking
    if frame > 1
        old_pos = inf(size(pos));
        iter = 1;
        
        %translation search
        while iter <= params.refinement_iterations && any(old_pos ~= pos)
            [xt_npca, xt_pca] = get_subwindow(im, pos, sz, currentScaleFactor);
            
            xt = feature_projection(xt_npca, xt_pca, projection_matrix, cos_window);
            xtf = fft2(xt);
            
            responsef = sum(hf_num .* xtf, 3) ./ (hf_den + params.lambda);
            
            % if we undersampled features, we want to interpolate the
            % response so it has the same size as the image patch
            if params.interpolate_response > 0
                if params.interpolate_response == 2
                    % use dynamic interp size
                    interp_sz = floor(size(y) * featureRatio * currentScaleFactor);
                end
                
                responsef = resizeDFT2(responsef, interp_sz);
            end
            
            response = ifft2(responsef, 'symmetric');
            
            [row, col] = find(response == max(response(:)), 1);
            disp_row = mod(row - 1 + floor((interp_sz(1)-1)/2), interp_sz(1)) - floor((interp_sz(1)-1)/2);
            disp_col = mod(col - 1 + floor((interp_sz(2)-1)/2), interp_sz(2)) - floor((interp_sz(2)-1)/2);
            
            switch params.interpolate_response
                case 0
                    translation_vec = round([disp_row, disp_col] * featureRatio * currentScaleFactor);
                case 1
                    translation_vec = round([disp_row, disp_col] * currentScaleFactor);
                case 2
                    translation_vec = [disp_row, disp_col];
            end
            
            old_pos = pos;
            pos = pos + translation_vec;
            
            iter = iter + 1;
        end
        
        %%%%%%%% verification with Siamese Networks %%%%%%%%
        if mod(frame, verify_param.gap) == 0
            im_color = imread([params.video_path s_frames{frame}]);
            
            % extract feature for tracking result in current frame
            last_rect = rect_position(frame - 1, :);
            current_sz  = [last_rect(4) last_rect(3)];
            current_box = [pos([2,1]) - current_sz([2,1])/2, current_sz([2,1])];
            
            input_roi = get_rois(current_box, verify_param.imageSz, im_color);
            if isempty(input_roi)
                score = -inf;
            else
                tfeat = ptav_extract_features(im_color, input_roi, verify_param.pixel_means, verify_param.imageSz);
                if isempty(tfeat)
                    score = -inf;
                else
                    % compute verification score
                    score = tfeat' * verify_param.firstframe_feat;
                    score = max(score(:));
                end
            end
%             fprintf('frame:%d, score: %f\n', frame, score);
            
            if score < verify_param.threshold
                verification_result = false;
            end
        end
        
        if ~ verification_result    % tracking result is unreliable
            verify_param.gap = 1;   % decrease verification interval
            % do detection
            [pos, det_score] = correct_tracker(im, pos, current_sz, verify_param);
            if det_score >= verify_param.det_threshold
            end
        else
            verify_param.gap = verify_param.gap_bk;
        end
        %%%%%%%%         end of verification         %%%%%%%%
        
        %scale search
        if params.number_of_scales > 0
            
            %create a new feature projection matrix
            [xs_pca, xs_npca] = get_scale_subwindow(im,pos,base_target_sz,currentScaleFactor*scaleSizeFactors,scale_model_sz);
            
            xs = feature_projection_scale(xs_npca,xs_pca,scale_basis,scale_window);
            xsf = fft(xs,[],2);
            
            scale_responsef = sum(sf_num .* xsf, 1) ./ (sf_den + params.lambda);
            
            interp_scale_response = ifft( resizeDFT(scale_responsef, params.number_of_interp_scales), 'symmetric');
            
            recovered_scale_index = find(interp_scale_response == max(interp_scale_response(:)), 1);
        
            %set the scale
            currentScaleFactor = currentScaleFactor * interpScaleFactors(recovered_scale_index);
            %adjust to make sure we are not to large or to small
            if currentScaleFactor < min_scale_factor
                currentScaleFactor = min_scale_factor;
            elseif currentScaleFactor > max_scale_factor
                currentScaleFactor = max_scale_factor;
            end
        end
    end
    
    %Compute coefficients for the tranlsation filter
    [xl_npca, xl_pca] = get_subwindow(im, pos, sz, currentScaleFactor);
    
    if frame == 1
        h_num_pca = xl_pca;
        h_num_npca = xl_npca;
        
        % set number of compressed dimensions to maximum if too many
        params.num_compressed_dim = min(params.num_compressed_dim, size(xl_pca, 2));
    else
        h_num_pca = (1 - params.interp_factor) * h_num_pca + params.interp_factor * xl_pca;
        h_num_npca = (1 - params.interp_factor) * h_num_npca + params.interp_factor * xl_npca;
    end;
    
    data_matrix = h_num_pca;
    
    [pca_basis, ~, ~] = svd(data_matrix' * data_matrix);
    projection_matrix = pca_basis(:, 1:params.num_compressed_dim);
    
    hf_proj = fft2(feature_projection(h_num_npca, h_num_pca, projection_matrix, cos_window));
    hf_num = bsxfun(@times, yf, conj(hf_proj));
    
    xlf = fft2(feature_projection(xl_npca, xl_pca, projection_matrix, cos_window));
    new_hf_den = sum(xlf .* conj(xlf), 3);
    
    if frame == 1
        hf_den = new_hf_den;
    else
        hf_den = (1 - params.interp_factor) * hf_den + params.interp_factor * new_hf_den;
    end
    
    %Compute coefficents for the scale filter
    if params.number_of_scales > 0
        
        %create a new feature projection matrix
        [xs_pca, xs_npca] = get_scale_subwindow(im, pos, base_target_sz, currentScaleFactor*scaleSizeFactors, scale_model_sz);
        
        if frame == 1
            s_num = xs_pca;
        else
            s_num = (1 - params.interp_factor) * s_num + params.interp_factor * xs_pca;
        end;
        
        bigY = s_num;
        bigY_den = xs_pca;
        
        if max_scale_dim
            [scale_basis, ~] = qr(bigY, 0);
            [scale_basis_den, ~] = qr(bigY_den, 0);
        else
            [U,~,~] = svd(bigY,'econ');
            scale_basis = U(:,1:s_num_compressed_dim);
        end
        scale_basis = scale_basis';
        
        % create the filter update coefficients
        sf_proj = fft(feature_projection_scale([],s_num,scale_basis,scale_window),[],2);
        sf_num = bsxfun(@times,ysf,conj(sf_proj));
        
        xs = feature_projection_scale(xs_npca,xs_pca,scale_basis_den',scale_window);
        xsf = fft(xs,[],2);
        new_sf_den = sum(xsf .* conj(xsf),1);
        
        if frame == 1
            sf_den = new_sf_den;
        else
            sf_den = (1 - params.interp_factor) * sf_den + params.interp_factor * new_sf_den;
        end;
    end
    
    target_sz = floor(base_target_sz * currentScaleFactor);
    
    % save position 
    rect_position(frame,:) = [pos([2,1]) - floor(target_sz([2,1])/2), target_sz([2,1])];
    
    % visualization
    if visualization == 1
        rect_position_vis = [pos([2,1]) - target_sz([2,1])/2, target_sz([2,1])];
        
        if frame == 1
            figure;
            im_handle = imshow(im, 'Border','tight', 'InitialMag', 100 + 100 * (length(im) < 500));
            rect_handle = rectangle('Position',rect_position_vis, 'EdgeColor','g', 'LineWidth', 3);
            text_handle = text(10, 10, int2str(frame), 'FontSize', 18);
            set(text_handle, 'color', [0 1 1]);
        else
            try
                set(im_handle, 'CData', im)
                set(rect_handle, 'Position', rect_position_vis)
                set(text_handle, 'string', int2str(frame));
                
            catch
                return
            end
        end
        
        drawnow
    end
end

results.type = 'rect';
results.res = rect_position;
end
