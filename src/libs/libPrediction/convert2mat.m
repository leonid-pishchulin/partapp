function [params, clusSize] = convert2mat(Ypred,clusCenters, clusters, nClus)

clusSize = zeros(nClus,1);%unique(Ypred);

nImg = size(Ypred,1);

nF = size(clusCenters,2);

params = zeros(nImg,2*nF);

for i = 1:nClus
    idxs = find(Ypred == i);
%     if (isempty(idxs))
%         disp('WARNING! Empty cluster!');
%     end
    clusSize(i) = length(idxs);
    if (clusSize(i) > 0)
        sigma = [0.0001 0.0001];
        if (size(clusters{i},1) == 1)
            sigma = ones(1,nF)*0.0001;
        else
            sigma = std(clusters{i});
        end
        par = [clusCenters(i,:) sigma];
        params(idxs,:) = repmat(par,length(idxs),1);
        if (clusSize(i) > 1)
            fprintf('clusidx: %d, nEx: %d\n', i, clusSize(i));
        end
    end
end
end