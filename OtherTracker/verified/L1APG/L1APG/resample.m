function [map_afnv, count] = resample(curr_samples, prob, afnv)
%% resample with respect to observation likelihood

nsamples = size(curr_samples, 1);
fallback_afnv = local_fallback_afnv(curr_samples, afnv);
prob(~isfinite(prob)) = 0;
prob(prob < 0) = 0;

prob_sum = sum(prob);

if(prob_sum <= 0)
    map_afnv = ones(nsamples, 1)*fallback_afnv;
    count = zeros(size(prob));
else
    prob = prob / prob_sum;
    count = round(nsamples * prob);
    count(~isfinite(count)) = 0;
    count(count < 0) = 0;

    map_afnv = [];
    for i=1:nsamples
        for j = 1:count(i)
            map_afnv = [map_afnv; curr_samples(i,:)];
        end
    end
    ns = sum(count); %number of resampled samples can be less or greater than nsamples
    if ~isfinite(ns)
        ns = 0;
    end
    map_afnv = [map_afnv; ones(max(nsamples-ns, 0), 1)*fallback_afnv]; %if less
    map_afnv = map_afnv(1:nsamples, :); %if more
end
end

function afnv = local_fallback_afnv(curr_samples, afnv)
if all(isfinite(afnv))
    return;
end

valid_rows = all(isfinite(curr_samples), 2);
if any(valid_rows)
    afnv = curr_samples(find(valid_rows, 1), :);
else
    afnv = [1 0 0 1 0 0];
end
end
