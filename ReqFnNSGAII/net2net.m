function [Pfit, P, bestP]  = net2net(gen, net, P, Y, para)
% Pfit = cell(length(P),1);
ACC = zeros(length(P),1);
netP = cell(length(P),1);
numH = zeros(length(P),1);

for p = 1:length(P)
    numH(p,1) = P{p}(1);
end
% [numH, idx] = sort(numH,'ascend');
% P = P(idx,:);

parfor p = 1: length(P)
    [netP{p}, ACC(p)] = evaFit_lbfgs(gen, p, P, numH, net, Y, para);
end

% regFact = 
% Pfit = 

idx_bestR = find(ACC==max(ACC));
idx_bestR_min_dep = numH(idx_bestR) == min(numH(idx_bestR));
bestR_min_dep = P(idx_bestR(idx_bestR_min_dep)); 
temp = cell2mat(bestR_min_dep);
[~, idxBest] = min(temp(:,2));
bestP = netP{idx_bestR(idxBest)};

Pfit = ones(size(ACC)) - ACC;

% Pfit = [Pfit MSE];


end