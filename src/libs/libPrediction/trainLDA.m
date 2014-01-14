function [acc, YpredTrain, B, prior] = trainLDA(X, labelsTrain)
fprintf('trainLDA()\n');
assert(isempty(find(labelsTrain <= 0)));

clusidxs = unique(labelsTrain);
nC = length(clusidxs);

%% prior on class occurence
prior = zeros(1,nC);
for i=1:nC
    prior(i) = length(find(labelsTrain == clusidxs(i)))/length(labelsTrain);
end

%% convert to classify label format
labels = zeros(size(labelsTrain,1),nC);
for i=1:size(labelsTrain,1)
    labels(i,find(clusidxs == labelsTrain(i))) = 1;
end

%% SLDA parameters
delta = 1e-3; % l2-norm constraint
stop = -30; % request 30 non-zero variables
maxiter = 250; % maximum number of iterations
Q = nC - 1; % request two discriminative directions

%% low dimensional space by sLDA
[B, ~] = slda(X, labels, delta, stop, Q, maxiter);
DC = X*B;

%% classify in low dim space
[YpredTrain, errTrain] = classify(DC, DC, labelsTrain, 'linear', prior);%linear
acc = 1 - errTrain;
fprintf('training acc: %2.1f %%\n', 100*acc);
end