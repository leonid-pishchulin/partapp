function model = pascal_train(cls, note, annolistPos, annolistNeg, numparts)

% model = pascal_train(cls, n, note)
% Train a model with 2*n components using the PASCAL dataset.
% note allows you to save a note with the trained model
% example: note = 'testing FRHOG (FRobnicated HOG)

% At every "checkpoint" in the training process we reset the 
% RNG's seed to a fixed value so that experimental results are 
% reproducible.
initrand();

if nargin < 3
  note = '';
end

if nargin < 5
  numparts = 8;
end

globals; 
[pos, neg, compidx] =  loadData(annolistPos,annolistNeg);

n = length(unique(compidx));

save([cachedir cls '_train_' VOCyear], 'pos', 'neg');

% spos = split(cls, pos, n); % split data by aspect ratio into n groups
spos = splitCompIdx(pos, n, compidx); % split data based on cluster ids

cachesize = 24000;
% leonid: 
maxneg = 200;
% maxneg = 239;
% maxneg = length(neg);

% train root filters using warped positives & random negatives
try
  load([cachedir cls '_lrsplit1']);
catch
  initrand();
  for i = 1:n
    % split data into two groups: left vs. right facing instances
    models{i} = initmodel(cls, spos{i}, note, 'N');
    % comment out in case of no mirroring
    % inds = lrsplit(models{i}, spos{i}, i);
%     models{i} = train(cls, models{i}, spos{i}(inds), neg, i, 1, 1, 1, ...
%                       cachesize, true, 0.7, false, ['lrsplit1_' num2str(i)]);
  end
  save([cachedir cls '_lrsplit1'], 'models');
end

% train root left vs. right facing root filters using latent detections
% and hard negatives
try
  load([cachedir cls '_lrsplit2']);
catch
  initrand();
  for i = 1:n
    fprintf('compidx: %d\n',n);
    % comment out in case of no mirroring
    % models{i} = lrmodel(models{i});
    models{i} = train(cls, models{i}, spos{i}, neg(1:maxneg), 0, 0, 4, 3, ...
                      cachesize, true, 0.7, false, ['lrsplit2_' num2str(i)]);
  end
  save([cachedir cls '_lrsplit2'], 'models');
end

% merge models and train using latent detections & hard negatives
try 
  load([cachedir cls '_mix']);
catch
  initrand();
  model = mergemodels(models);
  model = train(cls, model, pos, neg(1:maxneg), 0, 0, 1, 5, ...
                cachesize, true, 0.7, false, 'mix');
  save([cachedir cls '_mix'], 'model');
end

% add parts and update models using latent detections & hard negatives.
try 
  load([cachedir cls '_parts']);
catch
  initrand();
  % 1:n in case of no mirroring
  for i = 1:n
  % for i = 1:2:2*n
    model = model_addparts(model, model.start, i, i, numparts, [6 6]);
  end
  model = train(cls, model, pos, neg(1:maxneg), 0, 0, 8, 10, ...
                cachesize, true, 0.7, false, 'parts_1');
  model = train(cls, model, pos, neg, 0, 0, 1, 5, ...
                cachesize, true, 0.7, true, 'parts_2');
  save([cachedir cls '_parts'], 'model');
end

save([cachedir cls '_final'], 'model');
