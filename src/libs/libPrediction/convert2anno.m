function [anno, params] = convert2anno(Ypred,anno,clusCenters,clusters,nClus)

clusidxs = (1:nClus);%unique(Ypred);
clusSize = zeros(nClus,1);%unique(Ypred);

nImg = size(Ypred,1);
nF = size(clusCenters,2);
params = zeros(nImg,2*nF);

for i = 1:nClus
    idxs = find(Ypred == clusidxs(i));
    for j=1:length(idxs)
        anno(idxs(j)).annorect(1).silhouette.id = clusidxs(i)-1;%i-1;
    end
    
    par = [clusCenters(i,:) std(clusters{i})];
    params(idxs,:) = repmat(par,length(idxs),1);
    
    clusSize(i) = length(idxs);
    fprintf('clusidx: %d, nEx: %d\n', i, clusSize(i));
    
end
fprintf('\n');
end