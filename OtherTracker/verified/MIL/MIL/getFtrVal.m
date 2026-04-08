function samplesFtrVal = getFtrVal(img,samples,ftr)
% $Description:
%    -Compute the features of samples
% $Agruments
% Input;
%    -img: inpute image
%    -samples: sample templates.samples.sx:x coordinate vector, samples.sy:
%    y coordinate vector
%    -ftr: feature template. ftr.px,ftr.py,ftr.pw,ftr.ph,ftr.pwt
% Output:
%    -samplesFtrVal: size: M x N, where M is the number of features, N is
%    the number of samples
% $ History $
%   - Created by Kaihua Zhang, on April 22th, 2011
iH = integral(img);
sx = samples.sx;
sy = samples.sy;
px = ftr.px;
py = ftr.py;
pw = ftr.pw;
ph = ftr.ph;
pwt= ftr.pwt;

samplesFtrVal = FtrVal(iH,sx,sy,px,py,pw,ph,pwt); %feature without preprocessing
