function annolist = getRotNormData(pidx,annolist,saveTo)

fprintf('\ngetRotNormData()\n');
addpath ~/code/icps/

[~, parts] = getJointsParts22();

imgDir = [saveTo '/rotnorm'];
if (~exist(imgDir, 'dir'))
    mkdir(imgDir);
end

R = [0 1; -1 0];
R2 = [0 -1; 1 0];

for imgidx = 1:length(annolist)
    fprintf('.');
    img = imread(annolist(imgidx).image.name);
    points = annolist(imgidx).annorect(1).annopoints.point;

    annopoint_idxs_rot = parts(pidx+1).xaxis;
    p1_rot = getAnnopointById(points, annopoint_idxs_rot(1));
    p2_rot = getAnnopointById(points, annopoint_idxs_rot(2));
    
    if (~isempty(p1_rot) && ~isempty(p2_rot))
        %% compute rotation (only if axes are defined)
        xaxis = ([p2_rot.x p2_rot.y] - [p1_rot.x p1_rot.y]);
        xaxis = xaxis/norm(xaxis);
        xaxis = R * xaxis';
        yaxis = R2 * xaxis;
        
        rot = atan2(xaxis(2), xaxis(1));
        
        if (isnan(rot))
%             fprintf('WARNING!\n');
%             fprintf('rot is NAN for pidx %d in image %d\n', pidx, imgidx);
            rot = 0.0;
        end
        
        %% rotate and center image
        img_rot = imrotate(img, rot*180/pi);%, 'crop'
        [Y, X, ~] = size(img);
        img_c = [X/2, Y/2];
        [Y, X, ~] = size(img_rot);
        img_c_rot = [X/2, Y/2];
        
        %% rotate annotations
        for i=1:length(points)
            px = points(i).x;
            py = points(i).y;
            diff_px = px - img_c(1);
            diff_py = py - img_c(2);
            px_rot =  diff_px*xaxis(1) - diff_py*yaxis(1) + img_c_rot(1);
            py_rot =  -diff_px*xaxis(2) + diff_py*yaxis(2) + img_c_rot(2);
            points(i).x = px_rot;
            points(i).y = py_rot;
        end
        
    else
        img_rot = img;
    end
    
    annolist(imgidx).annorect(1).annopoints.point = points;
    
    %% save image    
    [~, n, e] = fileparts(annolist(imgidx).image.name);
    fname = [imgDir '/' n e];
    imwrite(img_rot, fname);
    annolist(imgidx).image.name = fname;
    if (~mod(imgidx, 100))
        fprintf(' %d/%d\n',imgidx,length(annolist));
    end
end

fprintf('\ndone!\n');

%% save annotations for visualization
fname = [imgDir '/pidx_' padZeros(num2str(pidx),4) '.idl'];
saveannotations(annolist, fname);
fname = [imgDir '/pidx_' padZeros(num2str(pidx),4) '.al'];
saveannotations(annolist, fname);