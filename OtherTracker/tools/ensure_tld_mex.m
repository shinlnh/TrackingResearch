function ensure_tld_mex(tld_root)
%ENSURE_TLD_MEX Build TLD mexw64 binaries needed on Windows.

if nargin < 1 || exist(tld_root, 'dir') ~= 7
    error('ensure_tld_mex:missingRoot', 'Missing TLD root: %s', tld_root);
end

mex_dir = fullfile(tld_root, 'mex_tld');
lk_mex = fullfile(mex_dir, ['lk.' mexext]);
if exist(lk_mex, 'file') ~= 0
    delete(lk_mex);
end

required = {'fern', 'linkagemex', 'bb_overlap', 'warp', 'distance'};
if has_required_mex(mex_dir, required)
    return;
end

old_dir = pwd;
cleanup_obj = onCleanup(@() cd(old_dir)); %#ok<NASGU>
cd(mex_dir);

common_flags = {'-O', 'CXXFLAGS=$CXXFLAGS /std:c++17'};

fprintf('Building TLD mex files in %s\n', mex_dir);
mex(common_flags{:}, '-c', 'tld.cpp');
mex(common_flags{:}, 'fern.cpp', 'tld.obj', '-output', 'fern');
mex(common_flags{:}, 'linkagemex.cpp', '-output', 'linkagemex');
mex(common_flags{:}, 'bb_overlap.cpp', '-output', 'bb_overlap');
mex(common_flags{:}, 'warp.cpp', '-output', 'warp');
mex(common_flags{:}, 'distance.cpp', '-output', 'distance');

if ~has_required_mex(mex_dir, required)
    error('ensure_tld_mex:buildFailed', 'Failed to build one or more TLD mex files in %s', mex_dir);
end
end

function ok = has_required_mex(mex_dir, required)
ok = true;
for i = 1:numel(required)
    if exist(fullfile(mex_dir, [required{i} '.' mexext]), 'file') == 0
        ok = false;
        return;
    end
end
end
