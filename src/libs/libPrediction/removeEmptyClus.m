function [YpredTrainNew, YpredTestNew] = removeEmptyClus(YpredTrain, YpredTest)

nClusTrain = length(unique(YpredTrain));
nClusTest = length(unique(YpredTest));
assert(nClusTrain >= nClusTest);
for i=1:nClusTest
    assert(ismember(YpredTest(i),YpredTrain));
end

YpredTrainNew = -1*ones(length(YpredTrain),1);
YpredTestNew = -1*ones(length(YpredTest),1);
idNew = 0;
for id=1:max(YpredTrain)
    idxs = find(YpredTrain == id);
    if (~isempty(idxs))
       idNew = idNew + 1;
       YpredTrainNew(idxs) = idNew;
       idxs2 = find(YpredTest == id);
       if (~isempty(idxs))
           YpredTestNew(idxs2) = idNew;
       end
    end
end

assert(isempty(find(YpredTrainNew < 0)));
assert(isempty(find(YpredTestNew < 0)));
end