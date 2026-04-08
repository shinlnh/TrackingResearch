function samplesFtrVal = getFtrVal_det(iH,samples,ftr,selector)
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
sx = samples.sx;
sy = samples.sy;
px = ftr.px(selector,:);
py = ftr.py(selector,:);
pw = ftr.pw(selector,:);
ph = ftr.ph(selector,:);
pwt= ftr.pwt(selector,:);

samplesFtrVal = FtrVal(iH,sx,sy,px,py,pw,ph,pwt); %feature without preprocessing
