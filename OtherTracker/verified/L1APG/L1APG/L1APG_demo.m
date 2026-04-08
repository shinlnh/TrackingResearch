
% clear;
% close all
% clc

% times  = 1; %operate times; to avoid overwriting previous saved tracking result in the .mat format
% title = 'car4';
% res_path='results\';
% 
% %% parameter setting for each sequence
% switch title
%     case 'car4'
%         fprefix		= '.\car4\';
%         fext		= 'jpg';    %Image format of the sequence
%         numzeros	= 4;		%number of digits for the frame index
%         start_frame = 1;		% first frame index to be tracked
%         nframes		= 600;		% number of frames to be tracked
%         %Initialization for the first frame. 
%         %Each column is a point indicating a corner of the target in the first image. 
%         %The 1st row is the y coordinate and the 2nd row is for x.
%         %Let [p1 p2 p3] be the three points, they are used to determine the affine parameters of the target, as following
%         %    p1(65,55)-----------p3(170,53)
%         %         | 				|		 
%         %         |     target      |
%         %         | 				|	        
%         %   p2(64,140)--------------
%         init_pos= [55,140,53;
%                    65,64,170];
%         sz_T =[12,15];      % size of template    
% end
% 
% %prepare the file name for each image
% s_frames = cell(nframes,1);
% nz	= strcat('%0',num2str(numzeros),'d'); %number of zeros in the name of image
% for t=1:nframes
%     image_no	= start_frame + (t-1);
%     id=sprintf(nz,image_no);
%     s_frames{t} = strcat(fprefix,id,'.',fext);
% end

function rect_result = L1APG_demo(video_path, video)

[init_pos, s_frames] = load_sequence(video_path, video);
sz_T =[12,15];

%% parameters setting for tracking
para.lambda = [0.2,0.001,10]; % lambda 1, lambda 2 for a_T and a_I respectively, lambda 3 for the L2 norm parameter
para.angle_threshold = 40;
para.Lip	= 8;
para.Maxit	= 5;
para.nT		= 10;%number of templates for the sparse representation
% para.rel_std_afnv = [0.03,0.0005,0.0005,0.03,1,1];%diviation of the sampling of particle filter
para.rel_std_afnv = [0.03,0.0001,0.0001,0.03,1,1];
para.n_sample	= 600;		%number of particles
para.sz_T		= sz_T;
para.init_pos	= init_pos;

%% main function for tracking
[tmp_result,~]  = L1TrackingBPR_APGup(s_frames, para);

for t = 1:numel(s_frames)
    map_afnv = tmp_result(:,t)';
    [rect, ~] = drawAffine(map_afnv, sz_T);
    rect_result(t,:) = rect;
end
% 
% for t = 1:numel(s_frames)
%     img_color	= imread(s_frames{t});
%     img_color	= double(img_color);
%     imshow(uint8(img_color));
%     text(5,10,num2str(t),'FontSize',18,'Color','r');
%     rectangle('position', rect_result(t,:), 'EdgeColor', 'r', 'LineWidth', 3);
%     drawnow;
% end
end