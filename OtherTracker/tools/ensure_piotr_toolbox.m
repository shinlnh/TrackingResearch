function ensure_piotr_toolbox(toolbox_root)
%ENSURE_PIOTR_TOOLBOX Build Piotr toolbox MEX files locally if bundled binaries are blocked.

if nargin < 1 || isempty(toolbox_root)
    error('ensure_piotr_toolbox:toolboxRootRequired', 'toolbox_root is required');
end

gradient_mex = fullfile(toolbox_root, 'channels', 'private', ['gradientMex.' mexext]);
compile_needed = exist(gradient_mex, 'file') ~= 3;

if ~compile_needed
    try
        I = single(rand(8, 8, 3));
        gradientMag(I, 0, 0, 0, 0); %#ok<NASGU>
    catch
        compile_needed = true;
    end
end

if ~compile_needed
    return;
end

old_dir = pwd;
cleanup_obj = onCleanup(@() cd(old_dir)); %#ok<NASGU>
cd(fullfile(toolbox_root, 'external'));
toolboxCompile();
rehash;
end
