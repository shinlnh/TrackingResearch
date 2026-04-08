function model = svmtrain(sample, label, varargin)
%SVMTRAIN Compatibility shim for legacy MEEM code on modern MATLAB.

opts.boxconstraint = 1;
opts.autoscale = true;
opts.options = struct();

if mod(numel(varargin), 2) ~= 0
    error('MEEM:svmtrainCompat', 'Arguments must be key/value pairs.');
end

for i = 1:2:numel(varargin)
    key = lower(string(varargin{i}));
    value = varargin{i + 1};
    switch key
        case "boxconstraint"
            opts.boxconstraint = value;
        case "autoscale"
            opts.autoscale = value;
        case "options"
            opts.options = value;
        otherwise
            error('MEEM:svmtrainCompat', 'Unsupported option: %s', key);
    end
end

sample = double(sample);
label = double(label(:));

standardize = true;
if islogical(opts.autoscale)
    standardize = opts.autoscale;
elseif ischar(opts.autoscale) || isstring(opts.autoscale)
    standardize = ~strcmpi(string(opts.autoscale), "false");
end

fitcsvm_args = {'KernelFunction', 'linear', 'Standardize', standardize};

if isscalar(opts.boxconstraint)
    fitcsvm_args = [fitcsvm_args, {'BoxConstraint', max(double(opts.boxconstraint), eps)}];
else
    weights = max(double(opts.boxconstraint(:)), eps);
    fitcsvm_args = [fitcsvm_args, {'BoxConstraint', 1, 'Weights', weights}];
end

if isstruct(opts.options) && isfield(opts.options, 'MaxIter') && ~isempty(opts.options.MaxIter)
    fitcsvm_args = [fitcsvm_args, {'IterationLimit', double(opts.options.MaxIter)}];
end

mdl = fitcsvm(sample, label, fitcsvm_args{:});
support_vectors = mdl.SupportVectors;

if isempty(support_vectors)
    alpha = zeros(0, 1);
else
    if exist('lsqminnorm', 'file') == 2
        alpha = lsqminnorm(support_vectors', mdl.Beta);
    else
        alpha = pinv(support_vectors') * mdl.Beta;
    end
end

model = struct();
model.Alpha = alpha;
model.SupportVectors = support_vectors;
model.SupportVectorIndices = find(mdl.IsSupportVector);
model.Bias = mdl.Bias;
model.Beta = mdl.Beta;
model.fitcsvm_model = mdl;
end
