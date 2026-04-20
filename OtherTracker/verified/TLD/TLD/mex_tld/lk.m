function output = lk(mode, imgI, imgJ, ptsI, ptsJ, level)
%LK MATLAB fallback for the original TLD Lucas-Kanade MEX.
%
% The original repo ships only Linux/MATLAB-32 binaries. On modern Windows
% MATLAB versions we use vision.PointTracker instead of the legacy OpenCV C API.

if nargin == 0
    disp('Lucas-Kanade');
    output = [];
    return;
end

if mode == 0
    output = [];
    return;
end

if mode ~= 2
    error('lk:unsupportedMode', 'Unsupported mode: %d', mode);
end

if nargin < 6 || isempty(level)
    level = 5;
end

if size(ptsI, 1) ~= 2 || size(ptsJ, 1) ~= 2 %#ok<NASGU>
    error('lk:invalidPoints', 'ptsI and ptsJ must be 2xN.');
end

imgI = to_gray_uint8(imgI);
imgJ = to_gray_uint8(imgJ);
nPts = size(ptsI, 2);
output = nan(4, nPts);

[ptsInit, validInitMask] = sanitize_points(ptsI.', size(imgI, 1), size(imgI, 2));
validInitIdx = find(validInitMask);
if isempty(validInitIdx)
    return;
end

tracker = vision.PointTracker( ...
    'NumPyramidLevels', max(1, min(8, level)), ...
    'BlockSize', [9 9], ...
    'MaxIterations', 20, ...
    'MaxBidirectionalError', inf);

initialize(tracker, ptsInit, imgI);
[forwardPts, isFound] = tracker(imgJ);
release(tracker);

if ~any(isFound)
    return;
end

foundLocalIdx = find(isFound);
[forwardPtsValid, validForwardMask] = sanitize_points(forwardPts(foundLocalIdx, :), size(imgJ, 1), size(imgJ, 2));
foundLocalIdx = foundLocalIdx(validForwardMask);
if isempty(foundLocalIdx)
    return;
end

backTracker = vision.PointTracker( ...
    'NumPyramidLevels', max(1, min(8, level)), ...
    'BlockSize', [9 9], ...
    'MaxIterations', 20, ...
    'MaxBidirectionalError', inf);

initialize(backTracker, forwardPtsValid, imgJ);
[backwardPtsFound, isBackFound] = backTracker(imgI);
release(backTracker);

for k = 1:numel(foundLocalIdx)
    localIdx = foundLocalIdx(k);
    idx = validInitIdx(localIdx);
    if ~isBackFound(k)
        continue;
    end

    ptForward = forwardPtsValid(k, :);
    ptBackward = backwardPtsFound(k, :);
    ptInit = ptsInit(localIdx, :);
    fb = norm(ptInit - ptBackward, 2);
    ncc = patch_ncc(imgI, imgJ, ptInit, ptForward);

    output(:, idx) = [ptForward(1); ptForward(2); fb; ncc];
end
end

function [points, validMask] = sanitize_points(points, height, width)
validMask = all(isfinite(points), 2);
if isempty(points)
    points = zeros(0, 2);
    return;
end

points = double(points);
if any(validMask)
    points(validMask, 1) = min(max(points(validMask, 1), 1), width);
    points(validMask, 2) = min(max(points(validMask, 2), 1), height);
end
points = points(validMask, :);
end

function img = to_gray_uint8(img)
if ndims(img) == 3
    img = rgb2gray(img);
end
if isa(img, 'uint8')
    return;
end
img = uint8(max(0, min(255, double(img))));
end

function value = patch_ncc(imgI, imgJ, ptI, ptJ)
patchI = extract_patch(imgI, ptI, 10);
patchJ = extract_patch(imgJ, ptJ, 10);

patchI = patchI(:) - mean(patchI(:));
patchJ = patchJ(:) - mean(patchJ(:));
denom = sqrt(sum(patchI .^ 2) * sum(patchJ .^ 2));
if denom <= eps
    value = 0;
else
    value = sum(patchI .* patchJ) / denom;
end
end

function patch = extract_patch(img, center, patchSize)
radius = (patchSize - 1) / 2;
[gridX, gridY] = meshgrid( ...
    center(1) + (-radius:radius), ...
    center(2) + (-radius:radius));
patch = interp2(double(img), gridX, gridY, 'linear', 0);
end
