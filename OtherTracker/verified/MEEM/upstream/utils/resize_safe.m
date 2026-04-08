function out = resize_safe(im, scale)
%RESIZE_SAFE Use the legacy MEX when available, else MATLAB imresize.

persistent mex_checked use_mex

if isempty(mex_checked)
    mex_checked = true;
    use_mex = false;
    try
        resize(uint8(zeros(2, 2, 'uint8')), 1);
        use_mex = true;
    catch err
        warning('MEEM:resizeSafeFallback', ...
            'resize.mexw64 unavailable, using imresize fallback: %s', err.message);
    end
end

if use_mex
    out = resize(im, scale);
else
    out = imresize(im, scale, 'bilinear');
end
end
