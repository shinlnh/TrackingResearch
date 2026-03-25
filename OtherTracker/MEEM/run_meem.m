function results = run_meem(seq, res_path, bSaveImage)
%RUN_MEEM OTB harness wrapper for the mirrored MEEM implementation.

if nargin < 1 || isempty(seq)
    error('MEEM:runWrapper', 'Sequence input is required.');
end

if nargin < 2
    res_path = ''; %#ok<NASGU>
end

if nargin < 3 || isempty(bSaveImage)
    bSaveImage = false; %#ok<NASGU>
end

global meem_sequence_name;
meem_sequence_name = seq.name;

image_dir = fileparts(seq.s_frames{1});
[~, ~, ext] = fileparts(seq.s_frames{1});
if startsWith(ext, '.')
    ext = ext(2:end);
end

results = MEEMTrack(image_dir, ext, false, seq.init_rect(1, :), 1, numel(seq.s_frames));
results.type = 'rect';
end
