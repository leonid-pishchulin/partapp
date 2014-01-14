function [clusidxs, data] = removeSmallClusters(clusidxs, data, minClusSize)
idxs = unique(clusidxs);

idxsRemove = [];

for i = 1:length(idxs)
    id = find(clusidxs == idxs(i));
    if (length(id) < minClusSize)
        idxsRemove = [idxsRemove; id];
    end
end

clusidxs(idxsRemove) = [];
data(idxsRemove) = [];

end