function rect_result = runTracker(video_path, video)
% RUN_TRACKER  is the external function of the tracker - does initialization and calls trackerMain

    %% Read params.txt
    params = readParams('params.txt');
	%% load video info
%     sequence_path = video_path;
	sequence_path = [video_path '/'];
    img_path = [sequence_path 'img/'];
%     %% Read files
%     text_files = dir([sequence_path '*_frames.txt']);
%     f = fopen([sequence_path text_files(1).name]);
%     frames = textscan(f, '%f,%f');
%     
%     start_frame = 1;
%     fclose(f);
    
    params.bb_VOT = dlmread([sequence_path '/groundtruth.txt']);
    region = params.bb_VOT(1,:);
    %%%%%%%%%%%%%%%%%%%%%%%%%
    % read all the frames in the 'imgs' subfolder
    dir_content = dir([sequence_path 'img/']);
    dir_content = dir_content(~[dir_content.isdir]);
    valid_ext = {'.jpg', '.jpeg', '.png', '.bmp'};
    is_valid = false(numel(dir_content), 1);
    for ii = 1:numel(dir_content)
        [~, ~, ext] = fileparts(dir_content(ii).name);
        is_valid(ii) = any(strcmpi(ext, valid_ext));
    end
    dir_content = dir_content(is_valid);
    [~, sort_idx] = sort({dir_content.name});
    dir_content = dir_content(sort_idx);

    n_imgs = numel(dir_content);
    img_files = cell(n_imgs, 1);
    for ii = 1:n_imgs
        img_files{ii} = dir_content(ii).name;
    end
    start_frame = 1;
    img_files(1:start_frame-1)=[];

    im = imread([img_path img_files{1}]);
    % is a grayscale sequence ?
    if(size(im,3)==1)
        params.grayscale_sequence = true;
    end

    params.img_files = img_files;
    params.img_path = img_path;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if(numel(region)==8)
        % polygon format
        [cx, cy, w, h] = getAxisAlignedBB(region);
    else
        x = region(1);
        y = region(2);
        w = region(3);
        h = region(4);
        cx = x+w/2;
        cy = y+h/2;
    end

    % init_pos is the centre of the initial bounding box
    params.init_pos = [cy cx];
    params.target_sz = round([h w]);
    [params, bg_area, fg_area, area_resize_factor] = initializeAllAreas(im, params);
	if params.visualization
		params.videoPlayer = vision.VideoPlayer('Position', [100 100 [size(im,2), size(im,1)]+30]);
	end
    % in runTracker we do not output anything
	params.fout = -1;
	% start the actual tracking
	results = trackerMain(params, im, bg_area, fg_area, area_resize_factor);
    rect_result = results.res;
end
