function rect_result = demo_ECO_HC_gpu(video_path, video)
% Run the hand-crafted ECO variant with GPU acceleration enabled.

[seq, ~] = load_video_info(video_path);
parallel.gpu.enableCUDAForwardCompatibility(true);

results = testing_ECO_HC(seq, [], false, struct('use_gpu', true, 'gpu_id', []));
rect_result = results.res;
end
