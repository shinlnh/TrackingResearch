clc
clear
close all

tracker = 'STC';

addpath(['./' tracker '/']);

benchmark_path = './LaSOTBenchmark/';
category = dir(benchmark_path);
category([1,2]) = [];

if ~exist(['./' tracker '_tracking_result/'], 'dir')
    mkdir(['./' tracker '_tracking_result/']);
end

for i = 1:numel(category)   % process each category
    tmp_cat = category(i).name;
    videos = dir([benchmark_path tmp_cat '/']);  % load sequences under each category
    videos([1,2]) = [];
    
    if ~exist(['./' tracker '_tracking_result/' tmp_cat], 'dir')
        mkdir(['./' tracker '_tracking_result/' tmp_cat]);
    end
    
    for k = 1:numel(videos)  % process each sequence for a category
        tmp_video = videos(k).name;
        tmp_video_path = [benchmark_path tmp_cat '/' tmp_video];
        
        fprintf('running %s on %s ... \n', tracker, tmp_video);
        
        % if this video has been processed, just ignore it
        if exist(['./' tracker '_tracking_result/' tmp_cat '/' tmp_video '_result.txt'], 'file')
            continue;
        end
        
        % run tracker using video's path and its name
        % rect_result stores the tracking results (bounding boxes)
        % note: we only revise the inputs and outputs of original trackers
        rect_result = demoSTC(tmp_video_path, tmp_video);
        
        % save results
        dlmwrite(['./' tracker '_tracking_result/' tmp_cat '/' tmp_video '_result.txt'], rect_result);
    end
end