function [img_files, pos, target_sz, ground_truth, new_video_path] = load_video_info(video_path, video)
%LOAD_VIDEO_INFO
%   Loads all the relevant information for the video in the given path:
%   the list of image files (cell array of strings), initial position
%   (1x2), target size (1x2), the ground truth information for precision
%   calculations (Nx2, for N frames), and the path where the images are
%   located. The ordering of coordinates and sizes is always [y, x].
%
%   Joao F. Henriques, 2014
%   http://www.isr.uc.pt/~henriques/


	%try to load ground truth from text file (Benchmark's format)
	filename = [video_path '/groundtruth.txt'];
	ground_truth = dlmread(filename);
	
	%set initial position and size
	target_sz = [ground_truth(1,4), ground_truth(1,3)];
	pos = [ground_truth(1,2), ground_truth(1,1)];
%     pos = [ground_truth(1,1), ground_truth(1,2)];
	
	%from now on, work in the subfolder where all the images are
	new_video_path = [video_path '/img/'];
	
	imgs = dir([new_video_path '*.jpg']);
	
    for i = 1:numel(imgs)
        img_files{i} = [new_video_path imgs(i).name];
    end
end

