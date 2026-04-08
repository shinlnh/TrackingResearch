
%   This script runs the original implementation of Background Aware Correlation Filters (BACF) for visual tracking.
%   the code is tested for Mac, Windows and Linux- you may need to compile
%   some of the mex files.
%   Paper is published in ICCV 2017- Italy
%   Some functions are borrowed from other papers (SRDCF, CCOT, KCF, etc)- and
%   their copyright belongs to the paper's authors.
%   copyright- Hamed Kiani (CMU, RI, 2017)

%   contact me: hamedkg@gmail.com
function rect_result = run_BACK_tracking(video_path, video)

new_video_path = [video_path '/'];
[seq, ~] = load_video_info(new_video_path);
seq.VidName = video;
seq.st_frame = 1;
seq.en_frame = seq.len;

%   Run BACF- main function
learning_rate = 0.013;  %   you can use different learning rate for different benchmarks.
results       = run_BACF_tracker(seq, new_video_path, learning_rate);
rect_result = results.res;
end