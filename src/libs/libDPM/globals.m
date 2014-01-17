% Set up global variables used throughout the code

% setup svm mex for context rescoring (if it's installed)
if exist('./svm_mex601') > 0
  addpath svm_mex601/bin;
  addpath svm_mex601/matlab;
end

% dataset to use
if exist('setVOCyear') == 1
  VOCyear = setVOCyear;
  clear('setVOCyear');
else
  VOCyear = '2007';
end

% directory for caching models, intermediate data, and results
% cachedir = ['/var/tmp/rbg/YOURPATH/' VOCyear '/'];

MCR_CACHE_ROOT2 = getenv('MCR_CACHE_ROOT2');
cachedir = [MCR_CACHE_ROOT2 '/']; 

% cachedir = ['/var/tmp/rbg/YOURPATH/' dirName '/'];


if exist(cachedir) == 0
  unix(['mkdir -p ' cachedir]);
  if exist([cachedir 'learnlog/']) == 0
    unix(['mkdir -p ' cachedir 'learnlog/']);
  end
end

% directory for LARGE temporary files created during training
% tmpdir = ['/var/tmp/rbg/YOURPATH/dat/' VOCyear '/'];
MCR_CACHE_ROOT = getenv('MCR_CACHE_ROOT');
tmpdir = [MCR_CACHE_ROOT '/']; 

if exist(tmpdir) == 0
  unix(['mkdir -p ' tmpdir]);
end

% should the tmpdir be cleaned after training a model?
cleantmpdir = true;

% directory with PASCAL VOC development kit and dataset
% VOCdevkit = ['/var/tmp/rbg/VOC' VOCyear '/VOCdevkit/'];
VOCdevkit = ['/BS/databases/pascal-voc/PASCAL2007/VOCdevkit/'];

