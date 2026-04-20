function tracked_bb = run_traca_lasot(class_dir, video)
%RUN_TRACA_LASOT Run TRACA on a single LaSOT sequence.

opt = struct();
opt.orth_lambda = 1000;
opt.finetune_iter = 10;
opt.finetune_rate = 1e-9;
opt.scale_ratio = 1.015;
opt.scale_variation = 3;
opt.val_min = 25;
opt.val_lambda = 50.0;
opt.output_sigma_factor = 0.05;
opt.lambda = 1.0;
opt.gamma = 0.025;
opt.redetect_n_frame = 50;
opt.redetect_eps = 0.7;
opt.redetect_gamma = 0.0025;
opt.visualization = 0;

ensure_traca_assets(pwd);
matconvnet_path = fullfile('..', 'DSiam', 'DSiam', 'matconvnet');
piotr_path = pwd;

try
    parallel.gpu.enableCUDAForwardCompatibility(true);
catch
end

[~, ~, tracked_positions] = tracker(class_dir, opt, matconvnet_path, piotr_path, video);
tracked_positions = double(tracked_positions);
tracked_bb = [ ...
    tracked_positions(:, 2) - tracked_positions(:, 4) / 2, ...
    tracked_positions(:, 1) - tracked_positions(:, 3) / 2, ...
    tracked_positions(:, 4), ...
    tracked_positions(:, 3)];
end
