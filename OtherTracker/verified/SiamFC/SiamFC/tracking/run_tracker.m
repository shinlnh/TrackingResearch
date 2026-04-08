function rect_results = run_tracker(video_path, video)
% RUN_TRACKER  is the external function of the tracker - does initialization and calls tracker.m
%     startup;
    %% Parameters that should have no effect on the result.
%     params.video = 'vot15_bag';
    
    params.video_path = video_path;
    params.video = video;
    params.visualization = 0;
    params.gpus = 1;
    %% Parameters that should be recorded.
    % params.foo = 'blah';
    %% Call the main tracking function
    rect_results = tracker(params);    
end