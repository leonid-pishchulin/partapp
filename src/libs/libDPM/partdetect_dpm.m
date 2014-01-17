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

function [det_all] = partdetect_dpm(imgidx,imgname,model,saveto,num_threads,bSaveScoregrid,rescale_factor,do_nms,num_rot,min_rot,max_rot,delta_scales)

fprintf('\npartdetect_dpm()\n');

%% init
if ischar(imgidx)
    imgidx = str2num(imgidx);
end

if ischar(model)
    fprintf('Load model from\n');
    fprintf('%s\n',model);
    m = load(model);
    model = m.model;
end

if (nargin < 5)
    num_threads = 1;
elseif (ischar(num_threads))
    num_threads = str2num(num_threads);
end

if (nargin < 6)
    bSaveScoregrid = 0;
elseif (ischar(bSaveScoregrid))
    bSaveScoregrid = str2num(bSaveScoregrid);
end

if (nargin < 7)
    rescale_factor = 1;
elseif (ischar(rescale_factor))
    rescale_factor = str2num(rescale_factor);
end

if (nargin < 8)
    do_nms = true;
elseif (ischar(do_nms))
    do_nms = str2num(do_nms);
end

if (nargin < 9)
    num_rot = 1;
elseif (ischar(num_rot))
    num_rot = str2num(num_rot);
end

if (nargin < 10)
    min_rot = -180;
elseif (ischar(min_rot))
    min_rot = str2num(min_rot);
end

if (nargin < 11)
    max_rot = -180;
elseif (ischar(max_rot))
    max_rot = str2num(max_rot);
end

if (nargin < 12)
    delta_scales = -1;
elseif (ischar(delta_scales))
    delta_scales = str2num(delta_scales);
end

step_size = (max_rot - min_rot)/num_rot;

THRESH = -3;

fprintf('imgname: %s\n',imgname);
fprintf('thresh = %f\n',THRESH);
fprintf('num_threads: %d\n', num_threads);
fprintf('bSaveScoregrid: %d\n', bSaveScoregrid);
fprintf('rescale_factor: %d\n', rescale_factor);
fprintf('do_nms: %d\n', do_nms);
fprintf('num_rot: %d\n', num_rot);
fprintf('delta_scales: %d\n', delta_scales);

if (isempty(saveto))
    bSaveDet = false;
else
    bSaveDet = true;
    if ~exist(saveto,'dir')
        mkdir(saveto);
    end
end

detections_al = [];
detections_al_comp = cell(0);
detections_mat = [];

img = imread(imgname);
[Y_orig, X_orig, Z_orig] = size(img);
    
if (rescale_factor ~= 1)
    img = imresize(img, rescale_factor, 'bicubic');
    [Y_orig, X_orig, Z_orig] = size(img);
end
    
%% run over rotations
for rotidx = 1:num_rot
        
    %% rotate image
    if (num_rot > 1)
        rotval = min_rot + step_size*(0.5 + rotidx - 1);
        img_rot = imrotate(img, rotval);
    else
        img_rot = img;
    end
        
    detections_al.image.name = imgname;
        
    %% detect
    if (do_nms)
        tic
        [det, all] = process(img_rot, model, THRESH, [], 0.0, num_threads, delta_scales);
        toc
        det_no_nms = det;
        all_no_nms = all;
    else
        [det, all, det_no_nms, all_no_nms] = process_nms(img_rot, model, THRESH, [], 0.0, num_threads);
        det = detNew;
        all = allNew;
        det_no_nms = detNew;
        all_no_nms = allNew;
    end
        
    %% map the detections back
    if (num_rot > 1)
        for ar=1:size(det_no_nms,1)
            x_c = 0.5*(det_no_nms(ar,1) + det_no_nms(ar,3));
            y_c = 0.5*(det_no_nms(ar,2) + det_no_nms(ar,4));
            w = det_no_nms(ar,3) - det_no_nms(ar,1);
            h = det_no_nms(ar,4) - det_no_nms(ar,2);
            [x_c_orig,y_c_orig] = rot_backproject(X_orig, Y_orig, X_rot, Y_rot, rotval, x_c, y_c);
            x1 = x_c_orig - w/2;
            x2 = x_c_orig + w/2;
            y1 = y_c_orig - h/2;
            y2 = y_c_orig + h/2;
            det_no_nms(ar,1:4) = [x1 y1 x2 y2];
        end
        
        for ar=1:size(det,1)
            x_c = 0.5*(det(ar,1) + det(ar,3));
            y_c = 0.5*(det(ar,2) + det(ar,4));
            w = det(ar,3) - det(ar,1);
            h = det(ar,4) - det(ar,2);
            [x_c_orig,y_c_orig] = rot_backproject(X_orig, Y_orig, X_rot, Y_rot, rotval, x_c, y_c);
            x1 = x_c_orig - w/2;
            x2 = x_c_orig + w/2;
            y1 = y_c_orig - h/2;
            y2 = y_c_orig + h/2;
            det(ar,1:4) = [x1 y1 x2 y2];
        end
    end
        
    %% rescale the detections to the original scale
    if (rescale_factor ~= 1)
        for ar=1:size(det,1)
            det(ar,1:4) = det(ar,1:4)/rescale_factor;
        end
        for ar=1:size(det_no_nms,1)
            det_no_nms(ar,1:4) = det_no_nms(ar,1:4)/rescale_factor;
        end
    end
        
    %% convert detections to annotation tool format
    for ar=1:size(det,1)
        detections_al.annorect(ar).x1 = det(ar,1);
        detections_al.annorect(ar).y1 = det(ar,2);
        detections_al.annorect(ar).x2 = det(ar,3);
        detections_al.annorect(ar).y2 = det(ar,4);
        detections_al.annorect(ar).score = det(ar,end);
        detections_al.annorect(ar).classID = all(ar,end-1);
        
        if (num_rot > 1)
            detections_al.annorect(ar).classID = rotidx;
        end
    end
        
    if (size(det_no_nms,1) > 0)
        detections_mat.det{rotidx} = [det_no_nms(:,1:4) all_no_nms(:,end-1:end)];
    else
        detections_mat.det{rotidx} = [];
        detections_al.annorect = [];
    end
    
    detections_al_comp{rotidx} = detections_al;
        
end
detections_mat.imgname = imgname;
fprintf('\n');
det_all = detections_mat;

%% save detections
% if (bSaveDet)
%     anno = detections_al_comp{1};
%     if (num_rot > 1)
%         for rotidx = 2:num_rot
%             annoNew = detections_al_comp{rotidx};
%             anno(end+1:end+length(annoNew)) = annoNew;
%         end
%     end
%     fname = [saveto '/det_' padZeros(num2str(pidx),4)];
%     saveannotations(anno, [fname '.al'], 1.0, 1.0, true); 
%     save([fname '.mat'], 'det_all');
% end

%% save scoregrids
if (bSaveScoregrid)  
    dpm2scoremap_smooth_cell(det_all, imgidx, saveto);
end
fprintf('\nDone\n');

end