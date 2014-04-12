function annolist = createCompTrainData(pidx,annolist,saveTo,clusterMode,alignRot,sc,nComp,minPartSize)

fprintf('\ncreateCompTrainData()\n');

if (nargin < 4)
    clusterMode = 1; % cluster rotation
end

if (nargin < 5)
    alignRot = false;
end

if (nargin < 6)
    sc = 1;
end

if (nargin < 7)
    nComp = 16;
end

if (nargin < 8)
    minPartSize = 30*sc;
end

%%
if (ischar(annolist))
    fprintf('Load annotations from\n');
    fprintf('%s\n',annolist);
    annolist = loadannotations(annolist);
end

fprintf('pidx: %d\n', pidx);
[~, parts] = getJointsParts22();

saveTo = [saveTo '/train_data'];
if (~exist(saveTo, 'dir'))
    mkdir(saveTo);
end

saveToCrops = [saveTo '/crops'];
if (~exist(saveToCrops, 'dir'))
    mkdir(saveToCrops);
end

%% normalize for rotation if needed
if (alignRot && ~clusterMode)
    annolist = getRotNormData(pidx,annolist,saveTo);
end

%% compute position/rotation features for clustering
clusidx = zeros(length(annolist),1);
if (clusterMode == 0)
    features = getPositionFeatures(annolist,parts,pidx);
elseif (clusterMode == 1)
    features = getRotationFeatures(annolist,parts,pidx);
elseif (clusterMode == 2)
    noassign = zeros(0);
    for imgidx = 1:length(annolist)
        if (~isfield(annolist(imgidx).annorect(1), 'silhouette') || ...
                isempty(annolist(imgidx).annorect(1).silhouette))
            noassign(end+1) = imgidx;
        else
            clusidx(imgidx) = annolist(imgidx).annorect(1).silhouette.id;
        end
    end
    annolist(noassign) = [];
    clusidx(noassign) = [];
    assert(~isempty(annolist));
    assert(max(clusidx) <= nComp);
else
    assert(false);
end

%% cluster the data based on features
if (clusterMode < 2)
    if (nComp > 1)
        fprintf('Run k-means\n');
        [clusidx, ~] = kmeans(features, nComp, 'replicates', 100, 'emptyaction', 'drop');
    else
        clusidx = ones(length(annolist),1);
    end
end
annolist(find(isnan(clusidx))) = [];
clusidx(find(isnan(clusidx))) = [];

fprintf('Total pos with complete annotations: %d\n',length(annolist));
annolistComp = cell(nComp,1);

%% assign training samples to clusters
smallPos = [];
fprintf('Assign images to clusters\n');
for imgidx = 1:length(annolist)
    compidx = clusidx(imgidx);
    annolist(imgidx).annorect.silhouette.id = compidx;
    fprintf('.');
    
    %% compute bbox around the part
    points = annolist(imgidx).annorect(1).annopoints.point;
    annopoint_idxs = parts(pidx+1).pos;
    p1 = getAnnopointById(points, annopoint_idxs(1));
    p2 = getAnnopointById(points, annopoint_idxs(2));
    x1 = min([p1.x p2.x]);
    x2 = max([p1.x p2.x]);
    y1 = min([p1.y p2.y]);
    y2 = max([p1.y p2.y]);
    
    x1 = x1 - 0.25*(x2 - x1);
    x2 = x2 + 0.25*(x2 - x1);
    y1 = y1 - 0.25*(y2 - y1);
    y2 = y2 + 0.25*(y2 - y1);
    if (y2 - y1) < minPartSize
        mid = (y2 + y1)/2;
        y2 = mid + minPartSize/2;
        y1 = mid - minPartSize/2;
    end
    if (x2 - x1) < minPartSize
        mid = (x2 + x1)/2;
        x2 = mid + minPartSize/2;
        x1 = mid - minPartSize/2;
    end
    img = imread(annolist(imgidx).image.name);
    [Y, X, ~] = size(img);
    x1 = max(x1,1);
    x2 = min(x2,X);
    y1 = max(y1,1);
    y2 = min(y2,Y);
    
    %% exclude examples on the image border
    if ((y2 - y1) < minPartSize || (x2 - x1) < minPartSize)
        smallPos(end+1) = imgidx;
    else
        annolist(imgidx).annorect(1).x1 = x1;
        annolist(imgidx).annorect(1).x2 = x2;
        annolist(imgidx).annorect(1).y1 = y1;
        annolist(imgidx).annorect(1).y2 = y2;
        
        if (isempty(annolistComp{compidx}))
            annolistComp{compidx} = annolist(imgidx);
        else
            annolistComp{compidx}(end+1) = annolist(imgidx);
        end
    end
    
    if (~mod(imgidx, 100))
        fprintf(' %d/%d\n',imgidx,length(annolist));
    end
end
fprintf('\ndone!\n');

fprintf('Exclude small pos\n');
annolist(smallPos) = [];
fprintf('Pos left: %d\n',length(annolist));

%% save annotations for visualization
fname = [saveTo '/pidx_' padZeros(num2str(pidx),4) '.idl'];
saveannotations(annolist, fname);

for compidx = 1:nComp
    fname = [saveTo '/pidx_' padZeros(num2str(pidx),4) '_tidx_' padZeros(num2str(compidx),4) '.idl'];
    saveannotations(annolistComp{compidx}, fname);
end