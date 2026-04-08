function selector = MilBoostClassifierUpdate(posx,negx,M,NumSel)
%%
%Description
%  - Select the most discriminative weak classifiers by MIL boosting method
% Arguments:
% Input:
%  - posx: contains information of positive samples
%  - negx: contains information of negative samples
%  - M: weak classifier pool size
%  - NumSel: number of selected weak classifiers
% Output:
%  - selector: size: 1xNumsel. indictors of selected weak classifiers
%-Changed by Kaihua Zhang, on May 18th, 2011
%%
[row,numpos] = size(posx.feature);
[row,numneg] = size(negx.feature);
Hpos = zeros(1,numpos);
Hneg = zeros(1,numneg);
count = 1;
selector = zeros(1,NumSel);
likl = zeros(1,M);
for s = 1:NumSel      
    for m = 1:M          
        Hp = Hpos+posx.pospred(m,:);
        pll= prod(1-sigmf(Hp,[1 0]),2);%all the positive instances are in one positive bags
        poslikl = sum(log(1-pll));
               
        Np = Hneg+negx.negpred(m,:);        
        nll = prod(sigmf(Np,[-1 0]),2);
        neglikl= sum(log(nll));      
        likl(m) = -poslikl - neglikl;%
    end        
        [likAsc,ind] = sort(likl,2);        
        for k=1:length(ind)
           if ~sum(selector == ind(k))
               selector(count) = ind(k);
               count = count + 1;
               break;
           end
        end          
        Hpos = Hpos + posx.pospred(selector(s),:);
        Hneg = Hneg + negx.negpred(selector(s),:); 
end