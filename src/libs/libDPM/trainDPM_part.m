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

function trainDPM_part(pidx,pos,neg,logDir,sc,cls,clusterMode,nComp,minPartSize)

fprintf('trainDPM_part()\n');

if (nargin < 5)
    sc = 2; 
end

if (nargin < 6)
    cls = 2; 
end

if (nargin < 7)
    clusterMode = 1; % rotation clustering
end

if (nargin < 8)
    nComp = 16; 
end

if (nargin < 9)
    minPartSize = 30*sc; 
end

if (nargin < 10)
    alignRot = false; 
end

if (nargin < 11)
    nParts = 8; 
end

if (~exist([logDir '/' cls], 'dir'))
    mkdir([logDir '/' cls]);
end
if (exist([logDir '/' cls '/params.log'], 'file') > 0)
    system(['rm ' ([logDir '/' cls '/params.log'])]);
end

if (isstruct(pos))
    fprintf('pos: %s\n', pos(1).image.name);
else
    fprintf('pos: %s\n', pos);
end

if (isstruct(neg))
    fprintf('neg: %s\n', neg(1).image.name);
else
    fprintf('neg: %s\n', pos);
    neg = loadannotations(neg);
end

diary([logDir '/' cls '/params.log']);
fprintf('pidx: %d\n', pidx);
fprintf('saveTo: %s\n', [logDir '/' cls]);
fprintf('clusterMode: %d\n', clusterMode);
fprintf('alignRot: %d\n', alignRot);
fprintf('sc: %d\n', sc);
fprintf('nComp: %d\n', nComp);
fprintf('nParts: %d\n', nParts);
diary off;

pos = createCompTrainData(pidx,pos,[logDir '/' cls],clusterMode,alignRot,sc,nComp,minPartSize);

pascal(cls, pos, neg, [logDir '/' cls], nParts);

end