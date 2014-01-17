function [detections_al, detections_mat] = detect(first,nImg,annolist,model,saveto,overlap,annolist_bbox)

fprintf('detect()\n');

if ischar(model)
    m = load(model);
    model = m.model;
end

if ischar(first)
    first = str2double(first);
end

if ischar(nImg)
    nImg = str2double(nImg);
end

% fprintf('first: %d\n', first);
% fprintf('nImg: %d\n', nImg);
% fprintf('saveto: %s\n', saveto);
if (nargin < 6)
    overlap = 0.0;
elseif ischar(overlap)
    overlap = str2double(overlap);
end

if (nargin < 7 || isempty(annolist_bbox))
    annotations_bbox = [];
    overlap = 0.0;
elseif (ischar(annolist_bbox))
    fprintf('Load bbox annotations from\n');
    fprintf('%s\n',annolist_bbox);
    annotations_bbox = loadannotations(annolist_bbox);
else
    annotations_bbox = annolist_bbox;
end
    
% fprintf('annolist_bbox: %s\n', annolist_bbox);
fprintf('overlap: %f\n', overlap);

if (ischar(annolist))
    fprintf('Load test annotations from\n');
    fprintf('%s\n',annolist);
    annotations = loadannotations(annolist);
else
    annotations = annolist;
end

detections_al = [];
detections_mat = cell(0,2);
if (isempty(saveto))
    bSave = false;
else
    bSave = true;
    detectDir = [saveto '/detections/'];
    if ~exist(detectDir,'file')
        mkdir(detectDir);
    end
end

THRESH = -3;
fprintf('Initian thresh = %f\n',THRESH);

last = first+nImg-1;
if last > length(annotations);
    last = length(annotations);
end

fprintf('Processing images\n');
for imgnum=1:last - first + 1
    fprintf('image %d\n',imgnum);
    
    imgname = annotations(imgnum+first-1).image.name;
    detections_al(imgnum).image.name = imgname;
    img = imread(imgname);
    
    if (~isempty(annotations_bbox))
        annorect = annotations_bbox(imgnum+first-1).annorect(1);
        bbox = [annorect.x1 annorect.y1 annorect.x2 annorect.y2];
%         [img, box] = croppos(img, bbox);
    else
        bbox = [];
    end
    
    [det, all] = process(img, model, THRESH, bbox, overlap);
    for ar=1:size(det,1)
        detections_al(imgnum).annorect(ar).x1 = det(ar,1);
        detections_al(imgnum).annorect(ar).y1 = det(ar,2);
        detections_al(imgnum).annorect(ar).x2 = det(ar,3);
        detections_al(imgnum).annorect(ar).y2 = det(ar,4);
        detections_al(imgnum).annorect(ar).score = det(ar,5);
    end
    if (size(det,1) > 0)
        detections_mat{imgnum,1} = [det(:,1:end-1) all(:,end-1:end)];
    end
    detections_mat{imgnum,2} = imgname;
end

if (bSave)
    fname = [detectDir '/imgidx-' num2str(first) '-' num2str(last)];
    saveannotations(detections_al, [fname '.al'], 1.0, 1.0, true);
    det_all = detections_mat;
    save([fname '.mat'], 'det_all');
end

end