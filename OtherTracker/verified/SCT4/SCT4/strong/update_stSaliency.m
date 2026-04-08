function [new_rf, result] = update_stSaliency(feature, mask, old_rf)

N_TREE = 1;
N_ADDTREE = 1;
MAX_TREE = 20;

[rf, ~] = init_pgrf(feature, mask, N_TREE, old_rf.mainTree{1,1});

new_rf.mainTree = cat(1, rf.mainTree(1:N_ADDTREE,1), old_rf.mainTree);

new_leaf2partial = rf.leaf2partialTree(:, 1:N_ADDTREE);
old_leaf2partial = old_rf.leaf2partialTree;
new_leaf2partial = sanitize_leaf_refs(new_leaf2partial, size(rf.partialTree, 1));
old_leaf2partial = sanitize_leaf_refs(old_leaf2partial, size(old_rf.partialTree, 1));

new_main_prob = rf.mainProb(:, 1:N_ADDTREE);
old_main_prob = old_rf.mainProb;
target_rows = max([size(new_leaf2partial, 1), size(old_leaf2partial, 1), ...
    size(new_main_prob, 1), size(old_main_prob, 1)]);

new_leaf2partial = pad_numeric_rows(new_leaf2partial, target_rows);
old_leaf2partial = pad_numeric_rows(old_leaf2partial, target_rows);
new_main_prob = pad_numeric_rows(new_main_prob, target_rows);
old_main_prob = pad_numeric_rows(old_main_prob, target_rows);

aa = max(new_leaf2partial(:));
if isempty(aa) || ~isfinite(aa)
    aa = 0;
end

new_rf.leaf2partialTree = ...
    cat(2, new_leaf2partial, old_leaf2partial + aa * (old_leaf2partial > 0));
if aa > 0 && ~isempty(rf.partialTree)
    add_count = min(aa, size(rf.partialTree, 1));
    new_rf.partialTree = cat(1, rf.partialTree(1:add_count,1), old_rf.partialTree);
    new_rf.subProb = cat(1, rf.subProb(1:add_count,1), old_rf.subProb);
else
    new_rf.partialTree = old_rf.partialTree;
    new_rf.subProb = old_rf.subProb;
end
new_rf.mainProb = cat(2, new_main_prob, old_main_prob);

if(size(new_rf.mainTree, 1) > MAX_TREE)
    new_rf.mainTree = new_rf.mainTree(1:MAX_TREE, 1);
    new_rf.leaf2partialTree = new_rf.leaf2partialTree(:,1:MAX_TREE);
    if(max(vec(new_rf.leaf2partialTree)) > 0)
        new_rf.partialTree = new_rf.partialTree(1:max(vec(new_rf.leaf2partialTree)), 1);
        new_rf.subProb = new_rf.subProb(1:size(new_rf.partialTree,1), 1);
    end
    new_rf.mainProb = new_rf.mainProb(:,1:MAX_TREE);
end

result = eval_pgrf(feature, new_rf);

end

function out = pad_numeric_rows(in, target_rows)
if nargin < 2 || isempty(target_rows) || target_rows < 0
    target_rows = size(in, 1);
end

if isempty(in)
    out = zeros(target_rows, 0);
    return;
end

out = in;
curr_rows = size(out, 1);
if curr_rows < target_rows
    out(curr_rows + 1:target_rows, :) = 0;
elseif curr_rows > target_rows
    out = out(1:target_rows, :);
end
end

function out = sanitize_leaf_refs(in, partial_count)
if isempty(in)
    out = in;
    return;
end

out = in;
out(~isfinite(out)) = 0;
out = round(out);
out(out < 0) = 0;
if nargin >= 2 && ~isempty(partial_count) && partial_count >= 0
    out(out > partial_count) = 0;
end
end
