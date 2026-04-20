function rect_result = demo_struck(video_path, video)

struct_config_file = './Struck/config.txt';
imgs = dir([video_path '/' video '/img/*.jpg']);
img_first = imread([video_path '/' video '/img/' imgs(1).name]);

width = size(img_first, 2);
height = size(img_first, 1);
startFrame = 1;
endFrame = numel(imgs);

sequence_path = fullfile(video_path, video);
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

result_file = fullfile('.', 'Struck', 'tmpStruckRes.txt');
if exist(result_file, 'file')
    delete(result_file);
end

if ispc
    setenv('PATH', ['C:\opencv\opencv_build\cuda\install\x64\vc17\bin;' getenv('PATH')]);
    candidates = { ...
        fullfile('.', 'Struck', 'struck.exe'), ...
        fullfile('.', 'Struck', 'build', 'bin', 'struck.exe'), ...
        fullfile('.', 'Struck', 'build', 'bin', 'Release', 'struck.exe'), ...
        fullfile('.', 'Struck', 'build-win64', 'bin', 'struck.exe'), ...
        fullfile('.', 'Struck', 'build-win64', 'bin', 'Release', 'struck.exe'), ...
        fullfile('.', 'Struck', 'build-win64', 'Release', 'struck.exe')};
else
    candidates = { ...
        fullfile('.', 'Struck', 'struck'), ...
        fullfile('.', 'Struck', 'build', 'bin', 'struck')};
end

status = 1;
for i = 1:numel(candidates)
    command = candidates{i};
    if ispc
        command = strrep(command, '/', '\');
    end
    if exist(command, 'file') ~= 2
        continue;
    end
    status = system(['"' command '"']);
    if status == 0
        break;
    end
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
