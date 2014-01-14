function predictFactors(idxMode, nClus, idxFactor, saveTo, poseletResponcesTestDir, torsoPosTestDir, annotationsTest, bUseOracle, nParts, torsoPartIdx)

fprintf('****************************************************************************\n');
fprintf('predictFactors()\n');

subsetDims = [];

bPredictPairwise = false;
bPredictUnary = false;

if (ischar(idxMode))
    idxMode = str2num(idxMode);
end

if (ischar(nClus))
    nClus = str2num(nClus);
end

if (ischar(idxFactor))
    idxFactor = str2num(idxFactor);
end

if (nargin < 8)
    bUseOracle = 0;
elseif (ischar(bUseOracle))
    bUseOracle = str2num(bUseOracle);
end

if (nargin < 9)
    nParts = 10;
elseif (ischar(nParts))
    nParts = str2num(nParts);
end

if (nargin < 10)
    torsoPartIdx = 4;
elseif (ischar(torsoPartIdx))
    torsoPartIdx = str2num(torsoPartIdx);    
end

%% prediction mode
switch idxMode
    case 0
        bPredictPairwise = true;
        dims = 7;
    case 1
        bPredictUnary = true;
        dims = 1;
    case 2
        bPredictUnary = true;
        dims = 2:3;
    otherwise
        assert(false);
end

%% save parameters to file
if (saveTo(end) == '\')
    saveTo = saveTo(1:end-1);
end

predDataDir = [saveTo '/pred_data'];
if (~exist(predDataDir, 'dir'))
    mkdir(predDataDir);
end

if (exist([predDataDir '/params.log'], 'file') > 0)
    system(['rm ' ([predDataDir '/params.log'])]);
end
diary([predDataDir '/params.log']);
fprintf('idxMode: %d\n', idxMode);
fprintf('idxFactor: %d\n', idxFactor);
fprintf('nClus: %d\n', nClus);
fprintf('nParts: %d\n', nParts);
fprintf('saveTo: %s\n', saveTo);
fprintf('poseletResponcesTestDir: %s\n', poseletResponcesTestDir);
fprintf('torsoPosTestDir: %s\n', torsoPosTestDir);
fprintf('bUseOracle: %d\n', bUseOracle);
fprintf('torsoPartIdx: %d\n', torsoPartIdx);
fprintf('annotationsTest: %s\n', annotationsTest);
diary off;

%% load annotations
fprintf('Loading annotations\n');
testData = loadannotations(annotationsTest);
nExTest = length(testData);

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
[poseletResponcesTest, poseParamsTest] = getPoseletResponces(testData, subsetDims, joints, poseletResponcesTestDir, torsoPosTestDir, torsoPartIdx, nExTest);

nPred = 0;
nJoints = length(joints);

%% predict pairwise factors
if bPredictPairwise
    fprintf('N mixtures/joint: %d\n\n',nClus);
    fprintf('dims: %d\n', dims);
    sumAccTest = 0;
    
    for j=1:nJoints
        
        if (idxFactor > -1 && idxFactor < nJoints && idxFactor ~= j-1)
            continue;
        end
        
        fprintf('\njidx: %d\n',j-1);
        
        %% load pose parameters
        fname = [predDataDir '/trainlist_clus_jidx_' num2str(joints(j).id) '_jointtypes_' num2str(nClus) '.mat'];
        load(fname, 'clusidx_train','data_train');
        
        %% assign labels to test clusters (only for evaluation)
        clusidxsTest = assignLabels(data_train,poseParamsTest{j}(:,dims),nClus,clusidx_train);
        
        %% load LDA reduction matrix B and mean/var of train data
        fname = [predDataDir '/trainlist_lda_jidx_' num2str(joints(j).id) '_jointtypes_' num2str(nClus) '.mat'];
        load(fname, 'mu','d','B','X_train_norm','prior');
                
        nPred = nPred + 1;
                
        %% predict parameters
        accTest = 1; cumAccTest = ones(1,nClus);
        
        uniqTest = unique(clusidxsTest);
        
        if (~bUseOracle && ~(length(uniqTest) == 1 && uniqTest == 1))
            X_test_norm = (poseletResponcesTest-ones(nExTest,1)*mu)./sqrt(ones(nExTest,1)*d);
            DC = X_train_norm*B;
            DC_test = X_test_norm*B;
            YpredTest = classify(DC_test, DC, clusidx_train, 'linear', prior);
            accTest = sum(clusidxsTest == YpredTest)/length(clusidxsTest);
        else
            %% oracle case: use ground truth assignment
            YpredTest = clusidxsTest;
        end
        
        sumAccTest = sumAccTest + accTest;
        
        %% save results
        clusidx_test = YpredTest - 1;
        saveto = [predDataDir '/testlist_pred_jidx_' num2str(joints(j).id) '_jointtypes_' num2str(nClus) '.mat'];
        save(saveto, 'clusidx_test');
        nPred = nPred + 1;
    end
    
    fprintf('\nDone!\n\n');
    fprintf('avg test acc: %2.1f %%.\n', 100*sumAccTest/nPred);
end

nPred = 0;

%% predict unary factors
if bPredictUnary
    nParts = length(parts);

    sumAccTrain = 0;
    sumAccTest = 0;
    
    poseParamsTrainPart = cell(nParts,1);
    poseParamsTestPart = cell(nParts,1);
    %% select parameters for single parts (1 - rot child, 3 4 - pos child)    
    for p=1:nParts
        for j=1:nJoints
            if (joints(j).child.id == parts(p).id)
                poseParamsTrainPart{p} = poseParamsTrain{j}(:,[1 3 4]);
                poseParamsTestPart{p} = poseParamsTest{j}(:,[1 3 4]);
                break;
            end
        end
    end
    
    poseParamsTrainPart{rootidx+1} = poseParamsTrain{rootidx/2}(:,[2 5 6]);
    poseParamsTestPart{rootidx+1} = poseParamsTest{rootidx/2}(:,[2 5 6]);
        
    accUn = 0;
    for j=1:nParts
        
        if (length(dims) == 2 && j - 1 == rootidx)
            continue;
        end
        
        if (idxFactor > -1 && idxFactor < nParts && idxFactor ~= j-1)
            continue;
        end
        
        fprintf('\npidx: %d\n',j-1);
        minClusSize = 0;
        
        %% cluster pose parameters
        [clusidxsTrain, clusidxsTest, clusCentersList, clustersList] = getClustersKMeansTestNN(poseParamsTrainPart{j}(1:nExTrain,:),poseParamsTestPart{j},nClus,dims,minClusSize);
        
        %% predict parameters
        accTrain = 1; accTest = 1;

        uniqTrain = unique(clusidxsTrain);
        uniqTest = unique(clusidxsTest);

        if (~bUseOracle && ~(length(uniqTrain) == 1 && length(uniqTest) == 1 && uniqTrain == 1 && uniqTest == 1))
            %% predict cluster assignment
            [YpredTest, accTrain, accTest, YpredTrain, cumAccTest] = classifyTestLDA(poseletResponcesTrain, poseletResponcesTest, clusidxsTrain, clusidxsTest);
        else
            %% oracle case: use ground truth assignment
            YpredTest = clusidxsTest;
            YpredTrain = clusidxsTrain;
        end
        
        %% get parameters for every image
        [params_train, ~] = convert2mat(YpredTrain,clusCentersList,clustersList,nClus);
        [params_test,  ~]  = convert2mat(YpredTest,clusCentersList,clustersList,nClus);

        [YpredTrain, YpredTest] = removeEmptyClus(YpredTrain, YpredTest);
        
        accUn = [accTrain, accTest];
        sumAccTrain = sumAccTrain + accTrain;
        sumAccTest  = sumAccTest  + accTest;

        %% save results
        if (idxMode == 1)
            rot_train = params_train;
            saveto = [predDataDir '/trainlist_pred_rot_pidx_' num2str(parts(j).id) '.mat'];
            save(saveto, 'rot_train');
            rot_test = params_test;
            saveto = [predDataDir '/testlist_pred_rot_pidx_' num2str(parts(j).id) '.mat'];
            save(saveto, 'rot_test');

            clusidx_train = YpredTrain - 1;
            saveto = [predDataDir '/trainlist_pred_rot_clus_pidx_' num2str(parts(j).id) '.mat'];
            save(saveto, 'clusidx_train');
            clusidx_test = YpredTest - 1;
            saveto = [predDataDir '/testlist_pred_rot_clus_pidx_' num2str(parts(j).id) '.mat'];
            save(saveto, 'clusidx_test');
            accUnRot = accUn;
            saveto = [predDataDir '/accUnRot_pidx_' num2str(parts(j).id) '.mat'];
            save(saveto, 'accUnRot')
            cumAccUnRot = cumAccTest;
            saveto = [predDataDir '/cumAccUnRot_pidx_' num2str(parts(j).id) '.mat'];
            save(saveto, 'cumAccUnRot')
        elseif (idxMode == 2)
            pos_train = params_train;
            saveto = [predDataDir '/trainlist_pred_pos_pidx_' num2str(parts(j).id) '.mat'];
            save(saveto, 'pos_train');
            pos_test = params_test;
            saveto = [predDataDir '/testlist_pred_pos_pidx_' num2str(parts(j).id) '.mat'];
            save(saveto, 'pos_test');
            
            clusidx_train = YpredTrain - 1;
            saveto = [predDataDir '/trainlist_pred_pos_clus_pidx_' num2str(parts(j).id) '.mat'];
            save(saveto, 'clusidx_train');
            clusidx_test = YpredTest - 1;
            saveto = [predDataDir '/testlist_pred_pos_clus_pidx_' num2str(parts(j).id) '.mat'];
            save(saveto, 'clusidx_test');
            accUnPos = accUn;
            saveto = [predDataDir '/accUnPos_pidx_' num2str(parts(j).id) '.mat'];
            save(saveto, 'accUnPos')
            cumAccUnPos = cumAccTest;
            saveto = [predDataDir '/cumAccUnPos_pidx_' num2str(parts(j).id) '.mat'];
            save(saveto, 'cumAccUnPos')
        end
        nPred = nPred + 1;
    end
    fprintf('\nDone!\n\n');
    fprintf('Result: avg training acc: %2.1f %%, avg test acc: %2.1f %%.\n', 100*sumAccTrain/nPred, 100*sumAccTest/nPred);
end
