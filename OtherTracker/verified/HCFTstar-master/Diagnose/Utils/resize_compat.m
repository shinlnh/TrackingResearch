function out = resize_compat(im, sz, mode)
%RESIZE_COMPAT Use legacy mexResize when available, otherwise imresize.

if nargin < 3 || isempty(mode)
    mode = 'auto';
end

try
    out = mexResize(im, sz, mode);
    return;
catch
end

interp = 'bilinear';
if strcmpi(mode, 'antialias')
    out = imresize(im, sz, interp, 'Antialiasing', true);
else
    out = imresize(im, sz, interp, 'Antialiasing', false);
end
