function results = run_Staple_benchmark(seq, res_path, bSaveImage)
%RUN_STAPLE_BENCHMARK Headless OTB entry point for Staple local benchmarking.

params = readParams('params.txt');
params.img_files = seq.s_frames;
params.img_path = '';
params.visualization = 0;
params.visualization_dbg = 0;

im = imread(params.img_files{1});
if size(im,3) == 1
    params.grayscale_sequence = true;
end

region = seq.init_rect;

if numel(region) == 8
    [cx, cy, w, h] = getAxisAlignedBB(region);
else
    x = region(1);
    y = region(2);
    w = region(3);
    h = region(4);
    cx = x + w / 2;
    cy = y + h / 2;
end

params.init_pos = [cy cx];
params.target_sz = round([h w]);

[params, bg_area, fg_area, area_resize_factor] = initializeAllAreas(im, params);

params.fout = -1;

results = trackerMain(params, im, bg_area, fg_area, area_resize_factor);
fclose('all');
end
