function sequences = load_otb_sequences(otb_root, sequence_file)
%LOAD_OTB_SEQUENCES Build OTB sequence structs from a dataset root.
%
% This mirrors the sequence handling in the OTB toolkit, but reads the
% actual dataset from an external root such as otb/otb100.

if nargin < 2 || isempty(sequence_file)
    sequence_file = fullfile(otb_root, 'SEQUENCES');
end

fid = fopen(sequence_file, 'r');
if fid == -1
    error('load_otb_sequences:openFailed', 'Failed to open sequence file: %s', sequence_file);
end

sequences = {};
cleanup_obj = onCleanup(@() fclose(fid)); %#ok<NASGU>

while true
    sequence_name = fgetl(fid);
    if ~ischar(sequence_name)
        break;
    end
    sequence_name = strtrim(sequence_name);
    if isempty(sequence_name)
        continue;
    end
    sequences{end + 1} = get_sequence(otb_root, sequence_name); %#ok<AGROW>
end

end

function seq = get_sequence(otb_root, sequence_name)
parts = strsplit(sequence_name, '-');
base_name = parts{1};

seq = struct();
seq.name = sequence_name;
seq.path = fullfile(otb_root, base_name);
seq.anno_file = fullfile(seq.path, annotation_filename(sequence_name));
seq.nz = 4;
seq.ext = 'jpg';
seq.startFrame = 1;
seq.annos = read_annotations(seq.anno_file);
seq.endFrame = size(seq.annos, 1);

switch sequence_name
    case 'Board'
        seq.nz = 5;
    case 'David'
        seq.startFrame = 300;
        seq.endFrame = 770;
    case 'Football1'
        seq.endFrame = 74;
    case 'Freeman3'
        seq.endFrame = 460;
    case 'Freeman4'
        seq.endFrame = 283;
    case 'BlurCar1'
        seq.startFrame = 247;
        seq.endFrame = 988;
    case 'BlurCar3'
        seq.startFrame = 3;
        seq.endFrame = 359;
    case 'BlurCar4'
        seq.startFrame = 18;
        seq.endFrame = 397;
    case 'Tiger1'
        seq.startFrame = 6;
        seq.endFrame = 354;
        seq.annos = seq.annos(seq.startFrame:seq.endFrame, :);
end

seq.len = seq.endFrame - seq.startFrame + 1;
seq.s_frames = cell(seq.len, 1);
frame_fmt = sprintf('%%0%dd.%s', seq.nz, seq.ext);
for i = 1:seq.len
    frame_num = seq.startFrame + i - 1;
    seq.s_frames{i} = fullfile(seq.path, 'img', sprintf(frame_fmt, frame_num));
end

end

function filename = annotation_filename(sequence_name)
filename = 'groundtruth_rect.txt';
parts = strsplit(sequence_name, '-');
if numel(parts) == 2
    filename = strrep(filename, '.', ['.' parts{2} '.']);
end
end

function annos = read_annotations(anno_file)
if ~exist(anno_file, 'file')
    error('load_otb_sequences:missingAnnotation', 'Missing annotation file: %s', anno_file);
end

annos = readmatrix(anno_file, 'FileType', 'text');
if isempty(annos) || size(annos, 2) < 4 || any(isnan(annos(:, 1:min(end, 4))), 'all')
    raw_text = fileread(anno_file);
    raw_text = strrep(raw_text, ',', ' ');
    raw_text = strrep(raw_text, sprintf('\t'), ' ');
    raw_text = regexprep(raw_text, '\r', ' ');
    values = sscanf(raw_text, '%f');
    if mod(numel(values), 4) ~= 0
        error('load_otb_sequences:badAnnotation', 'Failed to parse annotation file: %s', anno_file);
    end
    annos = reshape(values, 4, []).';
end

annos = annos(:, 1:4);
end
