function predictFactorsImg(idxMode, imgidx, loadFrom, saveTo, poseletResponcesTestDir, torsoPosTestDir, nParts, torsoPartIdx)

fprintf('****************************************************************************\n');
fprintf('predictFactorsImg()\n');

subsetDims = [];

bPredictPairwise = false;
bPredictUnary = false;

if (ischar(idxMode))
    idxMode = str2num(idxMode);
end

if (ischar(imgidx))
    imgidx = str2num(imgidx);
end

if (nargin < 7)
    nParts = 22;
elseif (ischar(nParts))
    nParts = str2num(nParts);
end

if (nargin < 8)
    torsoPartIdx = 10;
elseif (ischar(torsoPartIdx))
    torsoPartIdx = str2num(torsoPartIdx);    
end

%% prediction mode
switch idxMode
    case 0
        bPredictPairwise = true;
    case 1
        bPredictUnary = true;
    case 2
        bPredictUnary = true;
    otherwise
        assert(false);
end

%% save parameters to file
if (saveTo(end) == '\')
    saveTo = saveTo(1:end-1);
end

if (loadFrom(end) == '\')
    loadFrom = loadFrom(1:end-1);
end

predDataDir = loadFrom;
assert(exist(predDataDir, 'dir')>0);

predDataTestDir = saveTo;
if (~exist(predDataTestDir, 'dir'))
    mkdir(predDataTestDir);
end

if (exist([predDataTestDir '/params.log'], 'file') > 0)
    system(['rm ' ([predDataTestDir '/params.log'])]);
end
diary([predDataTestDir '/params.log']);
fprintf('idxMode: %d\n', idxMode);
fprintf('imgidx: %d\n', imgidx);
fprintf('nParts: %d\n', nParts);
fprintf('saveTo: %s\n', predDataTestDir);
fprintf('poseletResponcesTestDir: %s\n', poseletResponcesTestDir);
fprintf('torsoPosTestDir: %s\n', torsoPosTestDir);
fprintf('torsoPartIdx: %d\n', torsoPartIdx);
diary off;

if (nParts == 10)
    [joints, parts] = getJointsParts();
    rootidx = 4;
elseif (nParts == 22)
    [joints, parts] = getJointsParts22();
    rootidx = 10;
else
    fprintf('Wrong part number! Exiting.\n');
    assert(0);
end

%% get features and labels 
fprintf('\nGetting test data\n');
poseletResponcesTest = getPoseletResponcesTest(subsetDims, poseletResponcesTestDir, torsoPosTestDir, torsoPartIdx, imgidx);

nJoints = length(joints);

% load normalized train poselet features and mean/var
fname = [predDataDir '/train_poselet_features.mat'];
load(fname, 'mu','d','X_train_norm','firedPoseletsIdxs');

X_test = poseletResponcesTest(firedPoseletsIdxs)';
assert(size(X_test,2) == size(mu,2));
X_test_norm = (X_test-mu)./sqrt(d);

%% predict pairwise factors
if bPredictPairwise
    clusidx_test = zeros(nJoints,1);
    
    for j=1:nJoints
        % load clustered components
        fname = [predDataDir '/trainlist_gt_jidx_' num2str(joints(j).id) '.mat'];
        load(fname, 'clusidx_train','data_train','prior');
        
        % load LDA reduction matrix B
        fname = [predDataDir '/trainlist_lda_jidx_' num2str(joints(j).id) '.mat'];
        load(fname, 'B');
                
        % predict parameters
        uniqTrain = unique(clusidx_train);
        
        if (~(length(uniqTrain) == 1 && uniqTrain == 1))
            DC = X_train_norm*B;
            DC_test = X_test_norm*B;
            YpredTest = classify(DC_test, DC, clusidx_train, 'linear', prior);
        else
            YpredTest = 1;
        end
        clusidx_test(j) = YpredTest - 1;
    end
    
    %% save results
    saveto = [predDataTestDir '/testlist_pred_pwise_imgidx_' num2str(imgidx) '.mat'];
    save(saveto, 'clusidx_test');
end

%% predict unary factors
if bPredictUnary
    nParts = length(parts);
    
    if (idxMode == 1)
        unType = 'rot';
    else
        unType = 'pos';
    end
    
    fname = [predDataDir '/trainlist_gt_' unType '_pidx_' num2str(0) '.mat'];
    load(fname, 'clusidx_train','data_train','clusCentersList','clustersList','prior');
    clusidx_test = zeros(nParts,1);
    params_test = zeros(nParts,2*size(clusCentersList,2));
    
    for j=1:nParts
        
        if (idxMode == 2 && j - 1 == rootidx)
            continue;
        end
        
        % load clustered parameters
        fname = [predDataDir '/trainlist_gt_' unType '_pidx_' num2str(parts(j).id) '.mat'];
        load(fname, 'clusidx_train','data_train','clusCentersList','clustersList','prior');
        
        % load LDA reduction matrix B
        fname = [predDataDir '/trainlist_lda_' unType '_pidx_' num2str(parts(j).id) '.mat'];
        load(fname, 'B');
        
        %% predict parameters
        uniqTrain = unique(clusidx_train);
        
        if (~(length(uniqTrain) == 1 && uniqTrain == 1))
            DC = X_train_norm*B;
            DC_test = X_test_norm*B;
            YpredTest = classify(DC_test, DC, clusidx_train, 'linear', prior);
        else
            YpredTest = 1;
        end
        
        % get parameters for every image
        params_test(j,:)  = convert2matTest(YpredTest,clusCentersList,clustersList);
        % [YpredTrain, YpredTest] = removeEmptyClus(YpredTrain, YpredTest);
        clusidx_test(j) = YpredTest - 1;
    end
    
    %% save results
    if (idxMode == 1)
        unType = 'rot';
        rot_test = params_test;
        saveto = [predDataTestDir '/testlist_params_' unType '_imgidx_' num2str(imgidx) '.mat'];
        save(saveto, 'rot_test');
    else
        unType = 'pos';
        pos_test = params_test;
        saveto = [predDataTestDir '/testlist_params_' unType '_imgidx_' num2str(imgidx) '.mat'];
        save(saveto, 'pos_test');
    end
    saveto = [predDataTestDir '/testlist_pred_' unType '_imgidx_' num2str(imgidx) '.mat'];
    save(saveto, 'clusidx_test');
end
fprintf('Done\n');
