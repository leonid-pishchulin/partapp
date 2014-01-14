function clusidxsTest = assignLabels(dataTrain,dataTest,nClus,clusidxsTrain)

nTrain = size(dataTrain,1);
nTest = size(dataTest,1);

clusidxsTest = zeros(nTest,1);

for i=1:nTest
    dist = sqrt(sum((dataTrain - repmat(dataTest(i,:),nTrain,1)).^2,2));
    [~, idx] = min(dist);
    clusidxsTest(i) = clusidxsTrain(idx);
end

for i = 1:nClus
    fprintf('clusidx test: %d, nEx: %d\n', i, length(find(clusidxsTest == i)))
end

end