%% 
% Demo for MILTrack in paper 
% "Visual Tracking with Online Multiple Instance Learning"
% by Boris Babenko, Ming-Hsuan Yang, Serge Belongie ,CVPR 2009, Miami, Florida.
% Matlab code implemented by Kaihua Zhang, Dept.of Computing, HK PolyU.
% Email: zhkhua@gmail.com
% Note: the purpose of the code is only for better understanding the principle
% of the MILTrack. The results may be different from Boris's C++ code
% because of the different implmentations. Use at your own risk.
% Created by Kaihua Zhang, on April 22th, 2011
% Revised by Kaihua Zhang, on August 8th, 2012
%% 
function rect_result = MIL_main(tmp_video_path, tmp_video)
rand('state',0);
%% initialize image frame

img_data_path = [tmp_video_path '/img/'];
addpath(img_data_path);

% addpath('./data');
% load init.txt;
% initial tracker
% initstate = init;

gt = dlmread([tmp_video_path '/groundtruth.txt']);
initstate = gt(1, :);
% set path
% img_dir = dir('./data/*.png');
img_dir = dir([img_data_path '*.jpg']);
rect_result = zeros(numel(img_dir), 4);
%---------------------------
% number of frames
num = length(img_dir);
imgOrg = imread(img_dir(1).name);
img = double(imgOrg(:,:,1));
%% parameter settings
% number of negative samples
trparams.init_negnumtrain = 65;
% radical scope of positive samples
trparams.init_postrainrad = 4.0;
% object position [x y width height]
trparams.initstate = initstate;
% size of search window
trparams.srchwinsz = 25;
% classifier parameters
clfparams.width = trparams.initstate(3);
clfparams.height= trparams.initstate(4);
% feature parameters
% number of rectangle: 2-6
ftrparams.minNumRect = 2;
ftrparams.maxNumRect = 6;
% number of all weak classifiers,i.e,feature pool
M = 250;
% number of selected weak classifiers
numSel = 50; 
% learning rate parameter
lRate = 0.85;
%% feature distribution initialization
% mean of positive features
posx.mu = zeros(M,1);
negx.mu = zeros(M,1);
% standard deviation of positive features
posx.sig= ones(M,1);
negx.sig= ones(M,1);
%% compute feature template
[ftr.px,ftr.py,ftr.pw,ftr.ph,ftr.pwt] = HaarFtr(clfparams,ftrparams,M);
%% compute sample templates
posx.sampleImage = sampleImg(img,initstate,trparams.init_postrainrad,0,100000);
negx.sampleImage = sampleImg(img,initstate,2*trparams.srchwinsz,1.5*trparams.init_postrainrad,trparams.init_negnumtrain);
%% compute features
posx.feature = getFtrVal(img,posx.sampleImage,ftr);
negx.feature = getFtrVal(img,negx.sampleImage,ftr);
%% update distribution parameters
[posx.mu,posx.sig,negx.mu,negx.sig] = weakClassifierUpdate(posx,negx,posx.mu,posx.sig,negx.mu,negx.sig,lRate);
% update weak classifiers
posx.pospred = weakClassifier(posx,negx,posx,1:M);
negx.negpred = weakClassifier(posx,negx,negx,1:M);
%% train the MIL boosting classifier
selector = MilBoostClassifierUpdate(posx,negx,M,numSel);
%%
% save tracking results
tr = zeros(num,4);
tr(1,:) = initstate;
rect_result(1, :) = initstate;
% begin tracking
for i = 2:num
    imgOrg = imread(img_dir(i).name);
    img = double(imgOrg(:,:,1));
    detectx.sampleImage = sampleImg(img,initstate,trparams.srchwinsz,0,100000);
    iH = integral(img);
    detectx.feature = getFtrVal_det(iH,detectx.sampleImage,ftr,selector);
    %% compute the weak classifiers for all detected samples
    h = weakClassifier(posx,negx,detectx,selector);
    % compute the strong classifier
    H = sum(h);
    %% find the index of the sample with the maximal classifier response
    [c,index] = max(H);
    %% detection results
    x = detectx.sampleImage.sx(index);
    y = detectx.sampleImage.sy(index);
    width = detectx.sampleImage.sw(index);
    height = detectx.sampleImage.sh(index);
    initstate = [x y width height];
    tr(i,:) = initstate;
    %% show the tracking results
%     strcat(num2str(i-1),' iterations');    
%     imshow(uint8(imgOrg));
%     rectangle('Position',initstate,'LineWidth',4,'EdgeColor','r');
%     hold on;
%     text(5, 18, strcat('#',num2str(i)), 'Color','y', 'FontWeight','bold', 'FontSize',20);
%     title(num2str(i));
%     set(gca,'position',[0 0 1 1]); 
%     pause(0.001);  
    %% crop the positive and negative samples
    posx.sampleImage = sampleImg(img,initstate,trparams.init_postrainrad,0,100000);
    negx.sampleImage = sampleImg(img,initstate,1.5*trparams.srchwinsz,4+trparams.init_postrainrad,trparams.init_negnumtrain);
    %% compute the features of the samples    
    posx.feature = getFtrVal(img,posx.sampleImage,ftr);
    negx.feature = getFtrVal(img,negx.sampleImage,ftr);
    % update distribution parameters
    [posx.mu,posx.sig,negx.mu,negx.sig] = weakClassifierUpdate(posx,negx,posx.mu,posx.sig,negx.mu,negx.sig,lRate); 
    % update weak classifiers
    posx.pospred = weakClassifier(posx,negx,posx,1:M);
    negx.negpred = weakClassifier(posx,negx,negx,1:M);
    %% train the MIL boosting classifier
    selector = MilBoostClassifierUpdate(posx,negx,M,numSel);
    
    rect_result(i, :) = initstate;
end
end