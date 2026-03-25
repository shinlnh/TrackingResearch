function setup_paths()

pathstr = fileparts(mfilename('fullpath'));

addpath(pathstr);
run(fullfile(pathstr, 'matconvnet1.08', 'matlab', 'vl_setupnn.m'));
end
