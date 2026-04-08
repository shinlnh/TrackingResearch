function rect_results = demo_csr(video_path, video)

% set this to tracker directory
tracker_path = 'E:\A new benchmark\Evaluated Trackers\CSRDCF';
% add paths

addpath(tracker_path);
addpath(fullfile(tracker_path, 'mex'));
addpath(fullfile(tracker_path, 'utils'));
addpath(fullfile(tracker_path, 'features'));

visualize_tracker = 0;

% choose name of the VOT sequence
% sequence_name = 'ball1';    

% path to the folder with VOT sequences
% base_path = 'vot2016';
% base_path = fullfile(base_path, sequence_name);
% img_dir = dir(fullfile(base_path, '*.jpg'));

img_dir = dir([video_path '/img/*.jpg']);

% initialize bounding box - [x,y,width, height]
% gt = read_vot_regions(fullfile(base_path, 'groundtruth.txt'));
gt = dlmread([video_path '/groundtruth.txt']);

start_frame = 1;
time = zeros(numel(img_dir), 1);
n_tracked = 0;

if visualize_tracker
    figure(1); clf;
    set(gcf,'DoubleBuffer','on','MenuBar','none');
    axes('position', [0.00 0 1.00 1.0]);
end

frame = start_frame;
rect_results = zeros(numel(img_dir), 4);
% close all;
while frame <= numel(img_dir),  % tracking loop
	% read frame
%     impath = fullfile(base_path, img_dir(frame).name);

%     if mod(frame, 1) == 0
%         fprintf('processing %d/%d \n', frame, numel(img_dir));
%     end

    impath = [video_path '/img/' img_dir(frame).name];
    img = imread(impath);
    
	% initialize or track
	if frame == start_frame
        
        bb = gt(frame,:) + 1;  % add 1: ground-truth top-left corner is (0,0)
		tracker = create_csr_tracker(img, bb);
        bb = gt(frame,:);  % just to prevent error when plotting
        
        
    else
        [tracker, bb] = track_csr_tracker(tracker, img);
%         try
%             [tracker, bb] = track_csr_tracker(tracker, img);
%         catch
%             bb = rect_results(frame-1, :);
%         end
    end
    
    n_tracked = n_tracked + 1;
    
    % visualization and failure detection
    if visualize_tracker
        
        figure(1); if(size(img,3)<3), colormap gray; end
        imagesc(uint8(img))
        
        set(gca,'ytick',[]);  
        set(gca,'xtick',[]);
        
        hold on;
        rectangle('Position',bb,'LineWidth',3,'EdgeColor','g');

        text(15, 25, num2str(frame), 'Color','r', 'FontSize', 15, 'FontWeight', 'bold');
        
        
        hold off;
        if frame == start_frame
            truesize;
        end
        drawnow; 
    end
    rect_results(frame, :) = bb;   % bounding box result
    
    frame = frame + 1;

end

end  % endfunction
