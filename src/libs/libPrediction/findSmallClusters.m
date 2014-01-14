function [idxsOutliers] =  findSmallClusters(clusidxs, minClusSize)
idxsOutliers = [];

if (minClusSize <= 0)
    return;
end
clusidxsUnique = unique(clusidxs);
for i=1:length(clusidxsUnique)
    idxs = find(clusidxs == clusidxsUnique(i));
    if (length(idxs) < minClusSize)
        idxsOutliers = [idxsOutliers; idxs];
    end
end
end