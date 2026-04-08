%% Copyright (C) Jongwoo Lim and David Ross.
%% All rights reserved.

% clc;clear;close all;
% initialize variables
function rect_result = runtracker(video_path, video)
rand('state',0);  randn('state',0);

dataPath = [video_path '/img/'];

imgs = dir([dataPath '*.jpg']);
gt = dlmread([video_path '/groundtruth.txt']);
p = [gt(1,1)+floor(gt(1,3)/2) gt(1,2)+floor(gt(1,4)/2) gt(1,3) gt(1,4) 0.0];

% p = [188,192,110,130,0];
opt = struct('numsample',600, 'condenssig',0.25, 'ff',1, 'batchsize',5, 'affsig',[9,9,.05,.05,.005,.001]);

% load([title '.mat'],'data','datatitle','truepts');


param0 = [p(1), p(2), p(3)/32, p(5), p(4)/p(3), 0];
param0 = affparam2mat(param0);

frame = imread([video_path '/img/' imgs(1).name]);
if size(frame, 3) ~= 1
    frame = rgb2gray(frame);
end
frame = double(frame)/256;
% frame = double(data(:,:,1))/256;

if ~exist('opt','var')  opt = [];  end
if ~isfield(opt,'tmplsize')   opt.tmplsize = [32,32];  end
if ~isfield(opt,'numsample')  opt.numsample = 400;  end
if ~isfield(opt,'affsig')     opt.affsig = [4,4,.02,.02,.005,.001];  end
if ~isfield(opt,'condenssig') opt.condenssig = 0.01;  end

if ~isfield(opt,'maxbasis')   opt.maxbasis = 16;  end
if ~isfield(opt,'batchsize')  opt.batchsize = 5;  end
if ~isfield(opt,'errfunc')    opt.errfunc = 'L2';  end
if ~isfield(opt,'ff')         opt.ff = 1.0;  end
if ~isfield(opt,'minopt')
  opt.minopt = optimset; opt.minopt.MaxIter = 25; opt.minopt.Display='off';
end

tmpl.mean = warpimg(frame, param0, opt.tmplsize);
tmpl.basis = [];
tmpl.eigval = [];
tmpl.numsample = 0;
tmpl.reseig = 0;
sz = size(tmpl.mean);  N = sz(1)*sz(2);

param = [];
param.est = param0;
param.wimg = tmpl.mean;
% if (exist('truepts','var'))
%   npts = size(truepts,2);
%   aff0 = affparaminv(param.est);
%   pts0 = aff0([3,4,1;5,6,2]) * [truepts(:,:,1); ones(1,npts)];
%   pts = cat(3, pts0 + repmat(sz'/2,[1,npts]), truepts(:,:,1));
%   trackpts = zeros(size(truepts));
%   trackerr = zeros(1,npts); meanerr = zeros(1,npts);
% else
%   pts = [];
% end

% % draw initial track window
% drawopt = drawtrackresult([], 1, frame, tmpl, param, pts);
% drawopt.showcondens = 0;  drawopt.thcondens = 1/opt.numsample;

[rect0, ~] = calcRectCenter([32,32], param0);
% figure(1);
% imshow(frame);
% text(5,10,num2str(1),'FontSize',18,'Color','r');
% rectangle('position', rect0, 'EdgeColor', 'r', 'LineWidth', 3);

wimgs = [];
nframes = numel(imgs);
rect_position = zeros(nframes, 4);
rect_position(1,:) = rect0;
for f = 2:nframes
%   frame = double(data(:,:,f))/256;
  frame = imread([video_path '/img/' imgs(f).name]);
  if size(frame, 3) ~= 1
      frame = rgb2gray(frame);
  end
  frame = double(frame)/256;
  
  % do tracking
  param = estwarp_condens(frame, tmpl, param, opt);

  [rect_position(f,:), ~] = calcRectCenter([32,32], param.est);
  
  % do update
  wimgs = [wimgs, param.wimg(:)];
  if (size(wimgs,2) >= opt.batchsize)
    if (isfield(param,'coef'))
      ncoef = size(param.coef,2);
      recon = repmat(tmpl.mean(:),[1,ncoef]) + tmpl.basis * param.coef;
      [tmpl.basis, tmpl.eigval, tmpl.mean, tmpl.numsample] = ...
        sklm(wimgs, tmpl.basis, tmpl.eigval, tmpl.mean, tmpl.numsample, opt.ff);
      param.coef = tmpl.basis'*(recon - repmat(tmpl.mean(:),[1,ncoef]));
    else
      [tmpl.basis, tmpl.eigval, tmpl.mean, tmpl.numsample] = ...
        sklm(wimgs, tmpl.basis, tmpl.eigval, tmpl.mean, tmpl.numsample, opt.ff);
    end
%    wimgs = wimgs(:,2:end);
    wimgs = [];
    
    if (size(tmpl.basis,2) > opt.maxbasis)
      %tmpl.reseig = opt.ff^2 * tmpl.reseig + sum(tmpl.eigval(tmpl.maxbasis+1:end).^2);
      tmpl.reseig = opt.ff * tmpl.reseig + sum(tmpl.eigval(opt.maxbasis+1:end));
      tmpl.basis  = tmpl.basis(:,1:opt.maxbasis);
      tmpl.eigval = tmpl.eigval(1:opt.maxbasis);
      if (isfield(param,'coef'))
        param.coef = param.coef(1:opt.maxbasis,:);
      end
    end
  end
  
%   drawopt = drawtrackresult(drawopt, f, frame, tmpl, param, pts);
%   if f == 2
%       figure(1);clf;
%       set(gcf,'DoubleBuffer','on','MenuBar','none');
%       axes('position', [0.00 0 1.00 1.0]);
%   end
%   imshow(frame);
%   text(5,10,num2str(f),'FontSize',18,'Color','r');
%   rectangle('position', rect_position(f,:), 'EdgeColor', 'r', 'LineWidth', 3);
%   pause(1/10000);
end
rect_result = rect_position;
end
