function [clusidxsTrain, clusCenters, clusters] = getClus(dataTrain,nClus)

fprintf('Running kmeans, N clusters = %d\n',nClus);
warning off;
[clusidxsTrain, clusCenters] = kmeans(dataTrain, nClus, 'replicates', 100, 'emptyaction', 'drop');
warning on;

for i = 1:nClus
    fprintf('clusidx train: %d, nEx: %d\n', i, length(find(clusidxsTrain == i)))
end

clusters = cell(nClus,1);
for i = 1:nClus
    idxs = find(clusidxsTrain == i);
    clusters{i} = dataTrain(idxs,:);
end

end