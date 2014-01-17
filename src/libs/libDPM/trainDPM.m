function trainDPM(pidxs, pathToExp)

if (~isdeployed)
    addpath ../../../src/scripts/matlab/;
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

for pidx = pidxs 
    trainDPM_part(pidx,posUpscaled,neg,logDir,scale);
end
end