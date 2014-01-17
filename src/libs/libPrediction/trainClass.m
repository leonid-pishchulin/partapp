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

function trainClass(idxMode, nClus, idxFactor, saveTo, poseletResponcesTrainDir, torsoPosTrainDir, annotationsTrain, nParts, torsoPartIdx)

fprintf('****************************************************************************\n');
fprintf('trainClass()\n');

if (~isdeployed)
    addpath ../../scripts/matlab
end

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
    nParts = 22;
elseif (ischar(nParts))
    nParts = str2num(nParts);
end

if (nargin < 9)
    torsoPartIdx = 10;
elseif (ischar(torsoPartIdx))
    torsoPartIdx = str2num(torsoPartIdx);    
end

%% check whether matlab sees slda function
s = which('slda');
k = strfind(s, 'sparseLDA_v2');
if isempty(k)
    fprintf('slda not found! Aborting...\n');
    assert(false);
else
    which('slda')
    which('larsen')
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

predDataDir = saveTo;
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
fprintf('poseletResponcesTrainDir: %s\n', poseletResponcesTrainDir);
fprintf('torsoPosTrainDir: %s\n', torsoPosTrainDir);
fprintf('torsoPartIdx: %d\n', torsoPartIdx);
fprintf('annotationsTrain: %s\n', annotationsTrain);
diary off;

%% load annotations
fprintf('Loading annotations\n');
trainData = loadannotations(annotationsTrain);
nExTrain = length(trainData);

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

nPred = 0;
nJoints = length(joints);

idxsZeroTrain = sum(poseletResponcesTrain.^2,1);
firedPoseletsIdxs = find(idxsZeroTrain > 0);
X_train = poseletResponcesTrain(:,firedPoseletsIdxs);
[X_train_norm, mu, d] = normalize(X_train);

% save normalized train poselet features and mean/var
saveto = [predDataDir '/train_poselet_features.mat'];
save(saveto, 'mu','d','X_train_norm','firedPoseletsIdxs');

%% train pairwise classifiers
if bPredictPairwise
    fprintf('N mixtures/joint: %d\n\n',nClus);
    fprintf('dims: %d\n', dims);
    clusCentersList = cell(nJoints,1);
    clustersList = cell(nJoints,1);
    sumAccTrain = 0;
           
    for j=1:nJoints
        
        if (idxFactor > -1 && idxFactor < nJoints && idxFactor ~= j-1)
            continue;
        end
        
        fprintf('\njidx: %d\n',j-1);
        
        %% cluster parameters
        [clusidxsTrain, clusCentersList{j}, clustersList{j}] = getClus(poseParamsTrain{j}(:,dims),nClus);
        
        accTrain = 1;
        
        uniqTrain = unique(clusidxsTrain);
        
        %% train LDA classifier
        if (~(length(uniqTrain) == 1 && uniqTrain == 1))
            [accTrain, YpredTrain, B, prior] = trainLDA(X_train_norm, clusidxsTrain);
            sumAccTrain = sumAccTrain + accTrain;
            nPred = nPred + 1;
        else
            YpredTrain = clusidxsTrain;
            B = 1;
            prior = [1 zeros(1,nClus-1)];
        end

        %% save results
        % predicted labels on train set
        clusidx_train = YpredTrain - 1;
        saveto = [predDataDir '/trainlist_pred_jidx_' num2str(joints(j).id) '.mat'];
        save(saveto, 'clusidx_train');
        
        % gt cluster labels and clustered features on train set
        clusidx_train = clusidxsTrain;
        data_train = poseParamsTrain{j}(:,dims);
        saveto = [predDataDir '/trainlist_gt_jidx_' num2str(joints(j).id) '.mat'];
        save(saveto, 'clusidx_train','data_train','prior');
        
        % LDA reduction matrix B
        saveto = [predDataDir '/trainlist_lda_jidx_' num2str(joints(j).id) '.mat'];
        save(saveto, 'B');
        
    end
    fprintf('\nDone!\n\n');
    
    if (nPred == 0)
        nPred = 1;
        sumAccTrain = 1;
    end
    fprintf('avg training acc: %2.1f %%.\n', 100*sumAccTrain/nPred);
        
end

nPred = 0;

%% train unary classifiers
if bPredictUnary
    nParts = length(parts);

    sumAccTrain = 0;
    
    poseParamsTrainPart = cell(nParts,1);
    %% select parameters for single parts (1 - rot child, 3 4 - pos child)    
    for p=1:nParts
        for j=1:nJoints
            if (joints(j).child.id == parts(p).id)
                poseParamsTrainPart{p} = poseParamsTrain{j}(:,[1 3 4]);
                break;
            end
        end
    end
    
    poseParamsTrainPart{rootidx+1} = poseParamsTrain{rootidx/2}(:,[2 5 6]);
        
    for j=1:nParts
        
        if (length(dims) == 2 && j - 1 == rootidx)
            continue;
        end
        
        if (idxFactor > -1 && idxFactor < nParts && idxFactor ~= j-1)
            continue;
        end
        
        fprintf('\npidx: %d\n',j-1);
        
        %% cluster parameters
        [clusidxsTrain, clusCentersList, clustersList] = getClus(poseParamsTrainPart{j}(1:nExTrain,dims),nClus);
        
        accTrain = 1;

        uniqTrain = unique(clusidxsTrain);
        
        %% train LDA classifier
        [X_train_norm, mu, d] = normalize(poseletResponcesTrain);
        if (~(length(uniqTrain) == 1 && uniqTrain == 1))
            [accTrain, YpredTrain, B, prior] = trainLDA(X_train_norm, clusidxsTrain);
        else
            YpredTrain = clusidxsTrain;
            B = 1;
            prior = [1 zeros(1,nClus-1)];
        end
        
        [params_train, ~] = convert2mat(YpredTrain,clusCentersList,clustersList,nClus);

        % [YpredTrain, YpredTest] = removeEmptyClus(YpredTrain, YpredTest);
        sumAccTrain = sumAccTrain + accTrain;

        %% save results
        if (idxMode == 1)
            unType = 'rot';
        else
            unType = 'pos';
        end
        
        % predicted labels on train set
        clusidx_train = YpredTrain - 1;
        saveto = [predDataDir '/trainlist_pred_' unType '_pidx_' num2str(parts(j).id) '.mat'];
        save(saveto, 'clusidx_train');
        
        % gt cluster labels and clustered features on train set
        clusidx_train = clusidxsTrain;
        data_train = poseParamsTrainPart{j}(1:nExTrain,dims);
        saveto = [predDataDir '/trainlist_gt_' unType '_pidx_' num2str(parts(j).id) '.mat'];
        save(saveto, 'clusidx_train','data_train','clusCentersList','clustersList','prior');
        
        % LDA reduction matrix B
        saveto = [predDataDir '/trainlist_lda_' unType '_pidx_' num2str(parts(j).id) '.mat'];
        save(saveto, 'B');
        
        nPred = nPred + 1;
    end
    fprintf('\nDone!\n\n');
    fprintf('Result: avg training acc: %2.1f %%.\n', 100*sumAccTrain/nPred);
end
