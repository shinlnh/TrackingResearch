% Copyright 2011 Zdenek Kalal
%
% This file is part of TLD.
% 
% TLD is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% TLD is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with TLD.  If not, see <http://www.gnu.org/licenses/>.


function [BB1 shift] = bb_predict(BB0,pt0,pt1)

BB1 = [];
shift = [0; 0];

if isempty(BB0) || size(pt0, 2) < 2 || size(pt1, 2) < 2
    return;
end

valid = all(isfinite(pt0), 1) & all(isfinite(pt1), 1);
pt0 = pt0(:, valid);
pt1 = pt1(:, valid);

if size(pt0, 2) < 2 || size(pt1, 2) < 2
    return;
end

of  = pt1 - pt0;
dx  = median(of(1,:));
dy  = median(of(2,:));

d1  = pdist(pt0','euclidean');
d2  = pdist(pt1','euclidean');
ratio = d2 ./ max(d1, eps);
ratio = ratio(isfinite(ratio) & ratio > 0);

if isempty(ratio) || ~isfinite(dx) || ~isfinite(dy)
    return;
end

s   = median(ratio);

s1  = 0.5*(s-1)*bb_width(BB0);
s2  = 0.5*(s-1)*bb_height(BB0);

if ~isfinite(s1) || ~isfinite(s2)
    return;
end

BB1  = [BB0(1)-s1; BB0(2)-s2; BB0(3)+s1; BB0(4)+s2] + [dx; dy; dx; dy];
if ~all(isfinite(BB1)) || BB1(3) < BB1(1) || BB1(4) < BB1(2)
    BB1 = [];
    return;
end

shift = [s1; s2];
