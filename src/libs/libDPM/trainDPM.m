function trainDPM(pidxs, pathToExp, bIsHeadDet)

if (~isdeployed)
    addpath ../../../src/scripts/matlab/;
end

if (nargin < 3)
    bIsHeadDet = false;
end

n = matlabpool('size');
if (n == 0)
    matlabpool open 8;
end

scale = 2.0;
pos = [pathToExp '/images/LSP/train-h200-OC-add-flipped.al'];
upscaledDataDir = [pathToExp '/images/LSP/' num2str(scale) 'x/'];
posUpscaled = [upscaledDataDir  '/train-h200-OC-add-flipped-' num2str(scale) 'x.al'];
neg = [pathToExp '/images/dpm-neg/dpm-neg.al'];
logDir = [pathToExp '/log_dir/exp-lsp-local-app-model/dpm_model/'];

if (~exist(posUpscaled, 'file'))
    if (~exist(upscaledDataDir, 'dir'))
        mkdir(upscaledDataDir);
    end
    rescale_data(pos, pathToExp, upscaledDataDir, scale)
end

[p,n,e] = fileparts(neg);
neg = loadannotations(neg);

for imgidx = 1:length(neg)
    neg(imgidx).image.name = [pathToExp '/' neg(imgidx).image.name];
end

if (bIsHeadDet)
    assert(length(pidxs) == 1 && pidxs == 11)
    clusterMode = 2; % cluster viewpoint
    cls = 'head';
    nComp = 8;
    trainDPM_part(pidxs,posUpscaled,neg,logDir,scale,cls,clusterMode,nComp);
else
    for pidx = pidxs 
        cls = ['pidx_' padZeros(num2str(pidx),4)];
        trainDPM_part(pidx,posUpscaled,neg,logDir,scale,cls);
    end
end
end