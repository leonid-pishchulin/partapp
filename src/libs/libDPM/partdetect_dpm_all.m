% This file is part of the implementation of the human pose estimation model as described in the paper:
    
% Leonid Pishchulin, Micha Andriluka, Peter Gehler and Bernt Schiele
% Strong Appearance and Expressive Spatial Models for Human Pose Estimation
% IEEE International Conference on Computer Vision (ICCV'13), Sydney, Australia, December 2013

% Please cite the paper if you are using this code in your work.

% The code may be used free of charge for non-commercial and
% educational purposes, the only requirement is that this text is
% preserved within the derivative work. For any other purpose you
% must contact the authors for permission. This code may not be
% redistributed without permission from the authors.  

function partdetect_dpm_all(imgidx,imgname,num_parts,modelDir,saveto,num_threads,bSaveScoregrid,rescale_factor,do_nms,num_rot)

fprintf('\npartdetect_dpm_all()\n');

if ischar(num_parts)
    num_parts = str2num(num_parts);
end

for pidx = 0:num_parts-1
    modelSubDir = [modelDir '/pidx_' padZeros(num2str(pidx),4)];
    fprintf('modelSubDir: %s\n',modelSubDir);
    files = dir([modelSubDir '/*final.mat']);
    assert(length(files) == 1);
    modelFilename = [modelSubDir '/' files.name];
    savetoSubDir = [saveto '/pidx_' padZeros(num2str(pidx),4)];
    fprintf('savetoSubDir: %s\n',savetoSubDir);
    partdetect_dpm(imgidx,imgname,modelFilename,savetoSubDir,num_threads,bSaveScoregrid,rescale_factor,do_nms,num_rot);
end

end