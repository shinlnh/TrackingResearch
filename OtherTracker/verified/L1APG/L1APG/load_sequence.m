function [init_pos, s_frames] = load_sequence(video_path, video)
imgs = dir([video_path '/img/*.jpg']);
for i = 1:numel(imgs)
    s_frames{i} = [video_path '/img/' imgs(i).name];
end
gt = dlmread([video_path '/groundtruth.txt']);
x = gt(1,1);
y = gt(1,2);
w = gt(1,3);
h = gt(1,4);
init_pos = [y y+h y;
            x x x+w];
end