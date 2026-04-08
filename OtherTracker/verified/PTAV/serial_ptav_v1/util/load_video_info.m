function [img_files, pos, target_sz, ground_truth, video_path, init_box] = load_video_info(video)
	%full path to the video's files
	video_path = video;

	%try to load ground truth from text file (Benchmark's format)
	filename = [video_path 'groundtruth_rect.txt'];
    ground_truth = dlmread(filename);
    
    init_box = ground_truth(1, :);
	
	%set initial position and size
	target_sz = [ground_truth(1,4), ground_truth(1,3)];
	pos = [ground_truth(1,2), ground_truth(1,1)] + floor(target_sz/2);
	
	if size(ground_truth,1) == 1,
		%we have ground truth for the first frame only (initial position)
		ground_truth = [];
	else
		%store positions instead of boxes
		ground_truth = ground_truth(:,[2,1]) + ground_truth(:,[4,3]) / 2;
    end
	
	%from now on, work in the subfolder where all the images are
	video_path = [video_path 'img/'];
	
	%for these sequences, we must limit ourselves to a range of frames.
	%for all others, we just load all png/jpg files in the folder.
    video_name = regexp(video, '\', 'split');
    video_name = video_name{3};
	frames = {'David', 300, 770;
			  'Football1', 1, 74;
			  'Freeman3', 1, 460;
			  'Freeman4', 1, 283};
              %'Tiger1',6,354};
	
	idx = find(strcmpi(video_name, frames(:,1)));
	
	if isempty(idx),
		%general case, just list all images
		img_files = dir([video_path '*.png']);
		if isempty(img_files),
			img_files = dir([video_path '*.jpg']);
			assert(~isempty(img_files), 'No image files to load.')
		end
		img_files = sort({img_files.name});
	else
		%list specified frames. try png first, then jpg.
		if exist(sprintf('%s%04i.png', video_path, frames{idx,2}), 'file'),
			img_files = num2str((frames{idx,2} : frames{idx,3})', '%04i.png');
			
		elseif exist(sprintf('%s%04i.jpg', video_path, frames{idx,2}), 'file'),
			img_files = num2str((frames{idx,2} : frames{idx,3})', '%04i.jpg');
			
		else
			error('No image files to load.')
		end
		
		img_files = cellstr(img_files);
    end
end