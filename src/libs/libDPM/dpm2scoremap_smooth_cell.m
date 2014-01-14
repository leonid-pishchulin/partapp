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

function dpm2scoremap_smooth_cell(det_all, imgidx, outdir)

fprintf('\ndpm2scoremap_smooth_cell()\n');

if (~exist(outdir, 'dir'))
    assert(false);
end
% outdir = [outdir '/dpm_scoremaps_smooth'];

if (~exist(outdir, 'dir'))
    mkdir(outdir);
end

sigma = 5;

I = imread(det_all.imgname);
[Y, X, Z] = size(I);

nComp = length(det_all.det);
scoregrid = cell(nComp, 1);

for compidx = 1:nComp
    
    px = [];
    for ix = 1:X
        px = [px, repmat(ix, 1,Y)];
    end
    p_vec = [px;
        repmat(1:Y, 1, X)];
    
    scoregrid_flat = zeros(1,Y*X);
    scoregrid_comp = zeros(Y, X);
    det_score_list = ones(1,3);
    
    idx = 1;
    
    for ridx = 1:size(det_all.det{compidx},1)
        det_x = round(0.5*(det_all.det{compidx}(ridx,1) + det_all.det{compidx}(ridx,3)));
        det_y = round(0.5*(det_all.det{compidx}(ridx,2) + det_all.det{compidx}(ridx,4)));
        
        dpm_score = det_all.det{compidx}(ridx,6);
        
        if det_y >= 1 && det_y <= size(I, 1) && det_x >= 1 && det_x < size(I, 2)
            det_score_list(idx,:) = [det_x det_y dpm_score];
            idx = idx + 1;
        end
    end
    
    min_score = min(min(det_score_list(:,3)));
    det_score_list(:,3) = det_score_list(:,3) - min_score;
    
    for ridx = 1:size(det_score_list,1)
        det_score = det_score_list(ridx,3);
        c = [det_score_list(ridx,1);det_score_list(ridx,2)];
        d = sum(bsxfun(@minus, p_vec,c).^2, 1);
        scoregrid_flat = scoregrid_flat + (det_score * (1/(sqrt(2*pi)*sigma)) * exp(-0.5*d/sigma^2));
    end
    
    count = 1;
    for ix = 1:X
        for iy = 1:Y
            scoregrid_comp(iy,ix) = scoregrid_flat(count);
            count = count + 1;
        end
    end
    max_score = max(max(scoregrid_comp));
    if (max_score == 0)
        max_score = 1;
    end
    
    scoregrid_comp = scoregrid_comp/max_score;
    
    scoregrid{compidx} = scoregrid_comp;
end

output_filename = [outdir '/imgidx_' padZeros(num2str(imgidx), 4) '.mat'];
fprintf('saving %s\n', output_filename);
save(output_filename, 'scoregrid');
