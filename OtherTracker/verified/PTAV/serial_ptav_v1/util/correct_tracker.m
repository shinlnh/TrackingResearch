function [pos, score] = correct_tracker(im, pos, t_sz, verify_param)
global net;

im_color = im;
if size(im_color, 3) == 1
    im_color = cat(3, im_color, im_color, im_color); 
end

lamda         = verify_param.lamda;
rect          = [pos([2,1]) - t_sz([2,1])/2, t_sz([2,1])];

% get the surrounding region of current position of the target object
[new_im_color, new_rect] = get_surrounding(rect, im_color, lamda);    

% get candidates from the surrounding region (sliding window strategy)
object_cand_boxes = get_candidates(new_im_color, t_sz, verify_param.scale);
score = -inf;

if isempty(object_cand_boxes)
    return;
end

object_cand_boxes(:, 1) = object_cand_boxes(:, 1) + new_rect(1);
object_cand_boxes(:, 2) = object_cand_boxes(:, 2) + new_rect(2);

% obtain features for all candidate
input_roi   = get_rois(object_cand_boxes, verify_param.imageSz, im_color);
if isempty(input_roi)
    return;
end
tfeat        = ptav_extract_features(im_color, input_roi, verify_param.pixel_means, verify_param.imageSz);
if isempty(tfeat)
    return;
end

% compute verification score for each candidate within one batch
tmp_score    = tfeat' * verify_param.firstframe_feat;
[~, max_ids]  = sort(tmp_score, 'descend');

% get the candidate with the highest score
max_id       = max_ids(1);
m_score      = tmp_score(max_id);
tmp_box      = object_cand_boxes(max_id, :);
m_pos        = [tmp_box(2)+tmp_box(4)/2 tmp_box(1)+tmp_box(3)/2];

score = m_score;
if m_score >= verify_param.det_threshold
    pos = m_pos;
end

end
