 function rect_result = asla_tracker(video_path, video)

dataPath = [video_path '/img/'];

imgs = dir([dataPath '*.jpg']);
gt = dlmread([video_path '/groundtruth.txt']);

frameNum = numel(imgs);

p = [gt(1,1)+floor(gt(1,3)/2) gt(1,2)+floor(gt(1,4)/2) gt(1,3) gt(1,4) 0.0];
EXEMPLAR_NUM = 10;
opt = struct('numsample', 600, 'affsig', [4,4,0.01,0.0,0.005,0]); %

psize = [32, 32];
param0 = [p(1), p(2), p(3)/psize(1), p(5), p(4)/p(3), 0]'; %param0 = [px, py, sc, th,ratio,phi];   
param0 = affparam2mat(param0); 

%%
SC_param.mode = 2;
SC_param.lambda = 0.01;
% SC_param.lambda2 = 0.001; 
SC_param.pos = 'ture';
patch_size = 16;
step_size = 8;
[patch_idx, patch_num] = img2patch(psize, patch_size, step_size); 

%% initial tracking
result = [];
initial_tracking; 
TemplateDict = normalizeMat(exemplars_stack); 
patch_dict = reshape(TemplateDict(patch_idx,:), patch_size*patch_size, patch_num*EXEMPLAR_NUM); % patch dictionary
patch_dict = normalizeMat(patch_dict);
align_patch_longfeature = reshape(eye(patch_num),patch_num*patch_num,1); 
% sklm variables
tmpl.mean = mean(TemplateDict,2);     
tmpl.basis = [];                                        
tmpl.eigval = [];                                      
tmpl.numsample = 0;                                     
tmpl.warpimg = [];
[tmpl.basis, tmpl.eigval, tmpl.mean, tmpl.numsample] = sklm(TemplateDict, tmpl.basis, tmpl.eigval, tmpl.mean, tmpl.numsample);

%% tracking using proposed method
for f = begin+EXEMPLAR_NUM:frameNum
    if mod(f,200) == 0
        fprintf('processing %d/%d \n', f, frameNum);
    end
    frame = imread([video_path '/img/' imgs(f).name]);
    if size(frame,3)==3
        grayframe = rgb2gray(frame);
    else
        grayframe = frame;
        frame = double(frame)/255; 
    end  
    frame_img = double(grayframe)/255; % 
    % sampling    
    particles_geo = sampling(result(end,:), opt.numsample, opt.affsig);     
    candidates = warpimg(frame_img, affparam2mat(particles_geo), psize); 
    candidates = candidates.*(candidates>0); 
    [candidates,candidates_norm] = normalizeMat(reshape(candidates,psize(1)*psize(2), opt.numsample));
    % cropping patches
    particles_patches = candidates(patch_idx, :);
    particles_patches = reshape(particles_patches,patch_size*patch_size, patch_num*opt.numsample);
    candi_patch_data= normalizeMat(particles_patches); % l2-norm normalization    
    % sparse coding
    patch_coef = mexLasso(candi_patch_data, patch_dict, SC_param); 
    merge_coef = zeros(patch_num, patch_num*opt.numsample);       
    for i=1:EXEMPLAR_NUM
        merge_coef = merge_coef + abs(patch_coef((i-1)*patch_num+1:i*patch_num,:));
    end
    normalized_coef = merge_coef./(repmat(sum(merge_coef,1), patch_num, 1)+eps);
    % alignment-pooling
    patch_longfeatures = reshape(normalized_coef,patch_num*patch_num, opt.numsample);         
    % MAP inference
    sim_measure = sum(align_patch_longfeature'*patch_longfeatures,1) ;  
    conf = sim_measure;
    [sort_conf, sort_idx] = sort(conf,'descend');    
    best_idx = sort_idx(1);
    best_particle_geo = particles_geo(:, best_idx);       
    best_patch_coef = normalized_coef(:,(best_idx-1)*patch_num+1:best_idx*patch_num);

    %% template update
    tmpl.warpimg = [tmpl.warpimg,candidates(:,best_idx)];
    if size(tmpl.warpimg,2)==5
        [tmpl.basis, tmpl.eigval, tmpl.mean, tmpl.numsample] = sklm(tmpl.warpimg, tmpl.basis, tmpl.eigval, tmpl.mean, tmpl.numsample, 1);
        if  (size(tmpl.basis,2) > 10)          
            tmpl.basis  = tmpl.basis(:,1:10);   
            tmpl.eigval = tmpl.eigval(1:10);    
        end
        tmpl.warpimg = [];
        recon_coef = mexLasso((candidates(:,best_idx)-tmpl.mean), [tmpl.basis, eye(size(tmpl.basis,1)) ], SC_param); 
        recon = tmpl.basis*recon_coef(1:size(tmpl.basis,2))+tmpl.mean;
        % replace the template probabilistic
        random_weight = [0,(2).^(1:EXEMPLAR_NUM-1)];
        random_weight = cumsum(random_weight/sum(random_weight));
        random_num = rand(1,1);
        for i=2:EXEMPLAR_NUM-1
            if random_num>=random_weight(i-1)&random_num<random_weight(i)
                break;
            end
        end
        if random_num>=random_weight(EXEMPLAR_NUM-1)
            i = EXEMPLAR_NUM;
        end
        TemplateDict(:,i)=[];
        TemplateDict(:,EXEMPLAR_NUM) = normalizeMat(recon);
        patch_dict = reshape(TemplateDict(patch_idx,:), patch_size*patch_size, patch_num*EXEMPLAR_NUM); % patch dictionary
        patch_dict = normalizeMat(patch_dict);
    end
   
    %% draw result
    result = [result; affparam2mat(best_particle_geo)']; 
%     drawopt = drawtrackresult(drawopt, f, frame, psize, result(end,:)'); % 
%     imwrite(frame2im(getframe(gcf)),sprintf('result/%s/Result/%04d.png',title,f));   
end
rect_result = zeros(size(result,1), 4);

for i = 1:size(result,1)
    [rect_result(i,:), ~] = calcRectCenter(psize, result(i,:));
%     tmp_img = imread([video_path '/img/' imgs(i).name]);
%     if i == 1
%         figure('position',[100 100 size(tmp_img, 2) size(tmp_img, 1)]);clf;
%         set(gcf,'DoubleBuffer','on','MenuBar','none');
%         axes('position', [0.00 0 1.00 1.0]);
%     end
%     
%     imagesc(tmp_img);
%     set(gca,'ytick',[]);     
%     set(gca,'xtick',[]);
%     
%     rectangle('position', rect_result(i,:), 'LineWidth', 3, 'EdgeColor', 'r');
%     pause(0.0000001);
end

end