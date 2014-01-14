% This file is part of the implementation of the human pose estimation model as described in the paper:
    
% L. Pishchulin, M. Andriluka, P. Gehler and B. Schiele
% Strong Appearance and Expressive Spatial Models for Human Pose Estimation
% IEEE Conference on Computer Vision and Pattern Recognition (ICCV'13), Sydney, Australia, December 2013

% Please cite the paper if you are using this code in your work.

% The code may be used free of charge for non-commercial and
% educational purposes, the only requirement is that this text is
% preserved within the derivative work. For any other purpose you
% must contact the authors for permission. This code may not be
% redistributed without permission from the authors.  

function predict_factors(idxPredMode, nClus, idxPred, saveTo, poseletResponcesTrainDir, poseletResponcesTestDir, torsoPosTrainDir, torsoPosTestDir, bUseOracle, nParts, torsoPartIdx, annolistTrain, annolistTest)

fprintf('****************************************************************************\n');
fprintf('predict_factors()\n');

subsetDims = [];

bPredictPairwise = false;
bPredictUnary = false;

if (ischar(idxPredMode))
    idxPredMode = str2num(idxPredMode);
end

if (ischar(nClus))
    nClus = str2num(nClus);
end

if (ischar(idxPred))
    idxPred = str2num(idxPred);
end

if (nargin < 9)
    bUseOracle = 0;
elseif (ischar(bUseOracle))
    bUseOracle = str2num(bUseOracle);
end

if (nargin < 10)
    nParts = 10;
elseif (ischar(nParts))
    nParts = str2num(nParts);
end

if (nargin < 11)
    torsoPartIdx = 4;
elseif (ischar(torsoPartIdx))
    torsoPartIdx = str2num(torsoPartIdx);    
end

%% prediction mode
switch idxPredMode
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
fprintf('idxPredMode: %d\n', idxPredMode);
fprintf('idxPred: %d\n', idxPred);
fprintf('nClus: %d\n', nClus);
fprintf('nParts: %d\n', nParts);
fprintf('saveto_dir: %s\n', saveTo);
fprintf('poseletResponcesTrainDir: %s\n', poseletResponcesTrainDir);
fprintf('poseletResponcesTestDir: %s\n', poseletResponcesTestDir);
fprintf('torsoPosTrainDir: %s\n', torsoPosTrainDir);
fprintf('torsoPosTestDir: %s\n', torsoPosTestDir);
fprintf('bUseOracle: %d\n', bUseOracle);
fprintf('torsoPartIdx: %d\n', torsoPartIdx);
fprintf('annolistTrain: %s\n', annolistTrain);
fprintf('annolistTest: %s\n', annolistTest);
diary off;

%% load annotations
fprintf('Loading annotations\n');
trainData = loadannotations(annolistTrain);
testData = loadannotations(annolistTest);
nExTrain = length(trainData);
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
fprintf('\nGetting train data\n');
[poseletResponcesTrain, poseParamsTrain] = getPoseletResponces(trainData, subsetDims, joints, poseletResponcesTrainDir, torsoPosTrainDir, torsoPartIdx, nExTrain);
fprintf('\nGetting test data\n');
[poseletResponcesTest, poseParamsTest] = getPoseletResponces(testData, subsetDims, joints, poseletResponcesTestDir, torsoPosTestDir, torsoPartIdx, nExTest);
assert(size(poseletResponcesTest,2) == size(poseletResponcesTrain,2));

nPred = 0;
nJoints = length(joints);

%% predict pairwise factors
if bPredictPairwise
    fprintf('N mixtures/joint: %d\n\n',nClus);
    fprintf('dims: %d\n', dims);
    clusCentersList = cell(nJoints,1);
    clustersList = cell(nJoints,1);
    sumAccTrain = 0;
    sumAccTest = 0;
    accPwise = 0;
    
    for j=1:nJoints
        
        if (idxPred > -1 && idxPred < nJoints && idxPred ~= j-1)
            continue;
        end
        
        fprintf('\njidx: %d\n',j-1);
        minClusSize = 0;
        
        %% cluster pose parameters
        [clusidxsTrain, clusidxsTest, clusCentersList{j}, clustersList{j}] = getClustersKMeansTestNN(poseParamsTrain{j},poseParamsTest{j},nClus,dims,minClusSize);
        
        %% predict parameters
        accTrain = 1; accTest = 1; cumAccTest = ones(1,nClus);
        
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
        
        sumAccTrain = sumAccTrain + accTrain;
        sumAccTest = sumAccTest + accTest;
        
        %% get parameters for every image (TODO: save as *mat!!!)
        [annoPredTrain, params_train] = convert2anno(YpredTrain,trainData,clusCentersList{j},clustersList{j},nClus);
        [annoPredTest, params_test] = convert2anno(YpredTest,testData,clusCentersList{j},clustersList{j},nClus);
        
        %% save results
        saveto = [predDataDir '/trainlist_pred_jidx_' num2str(joints(j).id) '_jointtypes_' num2str(nClus) '.al'];
        saveannotations(annoPredTrain, saveto);
        saveto = [predDataDir '/testlist_pred_jidx_' num2str(joints(j).id) '_jointtypes_' num2str(nClus) '.al'];
        saveannotations(annoPredTest, saveto);
        rot_train = params_train;
        saveto = [predDataDir '/trainlist_pred_rot_params_jidx_' num2str(joints(j).id) '.mat'];
        save(saveto, 'rot_train');
        rot_test = params_test;
        saveto = [predDataDir '/testlist_pred_rot_params_jidx_' num2str(joints(j).id) '.mat'];
        save(saveto, 'rot_test');
        accPwise = [accTrain accTest];
        saveto = [predDataDir '/accPwise_jidx_' num2str(joints(j).id) '.mat'];
        save(saveto, 'accPwise');
        cumAccPwise = cumAccTest;
        saveto = [predDataDir '/cumAccPwise_jidx_' num2str(joints(j).id) '.mat'];
        save(saveto, 'cumAccPwise');
        nPred = nPred + 1;
    end
    
    fprintf('\nDone!\n\n');
    fprintf('Result: avg training acc: %2.1f %%, avg test acc: %2.1f %%.\n', 100*sumAccTrain/nPred, 100*sumAccTest/nPred);
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
                poseParamsTrainPart{p} = [poseParamsTrain{j}(:,[1 3 4])];
                poseParamsTestPart{p} = poseParamsTest{j}(:,[1 3 4]);
                break;
            end
        end
    end
    
    poseParamsTrainPart{rootidx+1} = [poseParamsTrain{rootidx/2}(:,[2 5 6])];
    poseParamsTestPart{rootidx+1} = poseParamsTest{rootidx/2}(:,[2 5 6]);
        
    accUn = 0;
    for j=1:nParts
        
        if (length(dims) == 2 && j - 1 == rootidx)
            continue;
        end
        
        if (idxPred > -1 && idxPred < nParts && idxPred ~= j-1)
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
        if (idxPredMode == 1)
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
        elseif (idxPredMode == 2)
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
