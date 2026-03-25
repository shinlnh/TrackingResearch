function results = ECCV2016_gpu_settings(seq, res_path, bSaveImage, parameters)

close all

s_frames = seq.s_frames;

cnn_params.nn_name = 'imagenet-vgg-m-2048.mat';
cnn_params.output_layer = [0 3 14];
cnn_params.downsample_factor = [4 2 1];
cnn_params.input_size_mode = 'adaptive';
cnn_params.input_size_scale = 1;
cnn_params.use_gpu = true;
cnn_params.gpu_id = 1;

params.t_features = {
    struct('getFeature', @get_cnn_layers, 'fparams', cnn_params), ...
};

params.t_global.normalize_power = 2;
params.t_global.normalize_size = true;
params.t_global.normalize_dim = true;

params.search_area_shape = 'square';
params.search_area_scale = 5.0;
params.min_image_sample_size = 200^2;
params.max_image_sample_size = 300^2;

params.refinement_iterations = 1;
params.newton_iterations = 5;

params.output_sigma_factor = 1/12;
params.learning_rate = 0.0075;
params.nSamples = 400;
params.sample_replace_strategy = 'lowest_prior';
params.lt_size = 0;

params.max_CG_iter = 5;
params.init_max_CG_iter = 100;
params.CG_tol = 1e-3;
params.CG_forgetting_rate = 10;
params.precond_data_param = 0.5;
params.precond_reg_param = 0.01;

params.use_reg_window = true;
params.reg_window_min = 1e-4;
params.reg_window_edge = 10e-3;
params.reg_window_power = 2;
params.reg_sparsity_threshold = 0.05;

params.interpolation_method = 'bicubic';
params.interpolation_bicubic_a = -0.75;
params.interpolation_centering = true;
params.interpolation_windowing = false;

params.number_of_scales = 5;
params.scale_step = 1.02;

params.visualization = 0;
params.debug = 0;

params.wsize = [seq.init_rect(1,4), seq.init_rect(1,3)];
params.init_pos = [seq.init_rect(1,2), seq.init_rect(1,1)] + floor(params.wsize/2);
params.s_frames = s_frames;

results = tracker(params);
