function rect_result = demo_struck(video_path, video)

struct_config_file = './Struck/config.txt';
imgs = dir([video_path '/' video '/img/*.jpg']);
img_first = imread([video_path '/' video '/img/' imgs(1).name]);

width = size(img_first, 2);
height = size(img_first, 1);
startFrame = 1;
endFrame = numel(imgs);

sequence_path = video_path;
sequence_name = video;

% sequence_path = 'sequences';
% sequence_name = 'car-1';
% width = 1280;
% height = 720;
% startFrame = 1;
% endFrame = 112;

content{1} = ['sequenceBasePath' ' = ' sequence_path];    % sequence
content{2} = ['sequenceName' ' = ' sequence_name];         % name
content{3} = ['frameWidth' ' = ' num2str(width)];            % width
content{4} = ['frameHeight' ' = ' num2str(height)];            % height
content{5} = ['startFrame' ' = ' num2str(startFrame)];            % height
content{6} = ['endFrame' ' = ' num2str(endFrame)];            % height

tmp_fid = fopen(struct_config_file, 'w');
for i = 1:numel(content)
    fprintf(tmp_fid, '%s\n', content{i});
end
fclose(tmp_fid);

command = './Struck/struck';
result_file = fullfile('.', 'Struck', 'tmpStruckRes.txt');
if exist(result_file, 'file')
    delete(result_file);
end

command = fullfile('.', 'Struck', 'struck');
if ispc
    command = strrep(command, '/', '\');
end
status = system(['"' command '"']);
if status ~= 0
    alt_command = fullfile('.', 'Struck', 'build', 'bin', 'struck');
    if ispc
        alt_command = strrep(alt_command, '/', '\');
    end
    status = system(['"' alt_command '"']);
end
if status ~= 0
    error('demo_struck:binaryFailed', 'Failed to execute Struck binary.');
end

rect_result = dlmread('./Struck/tmpStruckRes.txt');

% for i = 1:numel(imgs)
%     img = imread([video_path '/' video '/img/' imgs(i).name]);
%     
%     if i == 1
%         figure(1);clf;
%         set(gcf,'DoubleBuffer','on','MenuBar','none');
%         axes('position', [0.00 0 1.00 1.0]);
%     end
%     
%     imshow(img);
%     rectangle('position', rect_result(i, :), 'EdgeColor', 'r', 'LineWidth', 3);
%     text(20, 30, ['#' num2str(i)], 'FontSize', 18, 'Color', 'y');
%     set(gca,'ytick',[]);   
%     set(gca,'xtick',[]);
%     pause(1/1000);
% end

end
