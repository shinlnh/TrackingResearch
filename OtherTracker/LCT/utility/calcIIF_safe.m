function dst = calcIIF_safe(im, ksize, nbins)
%CALCIIF_SAFE Use the legacy MEX when available, else MATLAB fallback.

persistent mex_checked use_mex

if isempty(mex_checked)
    mex_checked = true;
    use_mex = false;
    try
        calcIIF(uint8(zeros(2, 2, 'uint8')), [1, 1], 1);
        use_mex = true;
    catch err
        warning('LCT:calcIIFSafeFallback', ...
            'calcIIF.mexw64 unavailable, using MATLAB fallback: %s', err.message);
    end
end

if use_mex
    dst = calcIIF(im, ksize, nbins);
    return;
end

src = uint8(im);
ksize = max(round(double(ksize(:)')), 1);
if numel(ksize) == 1
    ksize = [ksize, ksize];
end
nbins = max(1, round(double(nbins)));

kernel = ones(ksize(1), ksize(2), 'double') / prod(ksize);
step = max(1, floor(256 / nbins));
mask = false(size(src));
dst_acc = zeros(size(src), 'double');

for i = 0:(nbins - 1)
    low = i * step;
    high = min(i * step + step, 255);
    temp = src >= low & src <= high;
    mask = mask | temp;
    temp_blr = round(imfilter(double(temp) * 255, kernel, 'symmetric', 'same', 'conv'));
    dst_acc = dst_acc + double(mask) .* temp_blr;
end

dst = uint8(min(dst_acc, 255));
end
