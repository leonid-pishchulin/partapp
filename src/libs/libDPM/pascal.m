function [ap1, ap2] = pascal(cls, annolistPos, annolistNeg, saveTo, numparts)

% [ap1, ap2] = pascal(cls, n, note, annolistPos, annolistNeg, nPos, initType, doPascalTest)
% Train and score a model with 2*n components.
% annolistPos - annotations with positive examples
% annolistNeg - annotations with negative examples
% nPos - number of positive examples to sample from the positive set
% initType - initialization of DPM components
% initType == 0 - initialize by aspect ratio
% initType == 1 - initialize by clustering poses
% doPascalTest - test on Pascal VOC dataset

ap1 = [];
ap2 = [];

if ~isdeployed
    [~, temp] = unix('mktemp -d');
    setenv('MCR_CACHE_ROOT', temp(1:end -1));
    if (~exist(saveTo, 'dir'))
        unix(['mkdir ' saveTo]);
    end
    setenv('MCR_CACHE_ROOT2', saveTo);
end 

if nargin < 5
    numparts = 8;
end

globals;

note = '';
testyear = VOCyear;

% record a log of the training procedure
diary([cachedir cls '.log']);

model = pascal_train(cls, note, annolistPos, annolistNeg, numparts);

% lower threshold to get high recall
model.thresh = min(-1.1, model.thresh);
model = bboxpred_train(cls, testyear);
saveTo = [cachedir cls '_final'];
save(saveTo, 'model');
fprintf('Model saved to %s\n',saveTo);

% remove dat file if configured to do so
if cleantmpdir
  unix(['rm ' tmpdir cls '.dat']);
end
