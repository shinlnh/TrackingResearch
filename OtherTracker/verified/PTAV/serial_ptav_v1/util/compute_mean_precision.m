% clc
% clear
% close all
% 
% path = 'E:\TC128\';
% 
% videos = dir(path);
% videos([1,2]) = [];
% pp = zeros(1, numel(videos));
% for i = 1:numel(videos)
%     video = videos(i).name;
%     gt_rects = dlmread([path video '\' video '_gt.txt']);
%     res_rects = dlmread(['results\' video '.txt']);
%     
%     positions = gt_rects;
%     tmp_pos    = positions(:, [2, 1]) + positions(:, [4, 3])/2;
%     
%     ground_truth = res_rects(:,[2,1]) + res_rects(:,[4,3]) / 2;
%     
%     precisions = precision_plot(tmp_pos, ground_truth);
%     
%     pp(i) = precisions(20);
% end
% 
% mean_pp = mean(pp);
% 
% fprintf('Mean precisions at 20 pixels: %f\n', mean_pp);


clc
clear
close all

path = 'E:\TrackingBenchmark15\';

videos = dir(path);
videos([1,2]) = [];
dp = zeros(1, numel(videos));
op = dp;
cle = dp;
for i = 1:numel(videos)
    video = videos(i).name;
    gt_rects = dlmread([path video '\groundtruth_rect.txt']);
    res_rects = dlmread(['results\' video '.txt']);
    
%     if strcmp(video, 'Tiger1')
%         gt_rects = gt_rects(6:end, :);
%         res_rects = res_rects(6:end, :);
%     end

    if i == 35
        res_rects(size(gt_rects, 1)+1:end, :) = [];
    end
    
    [dp(i), op(i), cle(i)] = compute_performance_measures(res_rects, gt_rects);
end

mean_dp = mean(dp);
mean_op = mean(op);
mean_cle = mean(cle);

fprintf('Mean distance precisions at 20 pixels: %f\n', mean_dp);
fprintf('Mean overlap precisions at 0.5: %f\n', mean_op);
fprintf('Mean average center location error: %f\n', mean_cle);