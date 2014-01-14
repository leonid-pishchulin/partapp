function params = convert2matTest(Ypred,clusCenters,clusters)

nF = size(clusCenters,2);

if (size(clusters{Ypred},1) == 1)
    sigma = ones(1,nF)*0.0001;
else
    sigma = std(clusters{Ypred});
end
params = [clusCenters(Ypred,:) sigma];

end