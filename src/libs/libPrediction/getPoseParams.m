function result = getPoseParams(data, joint)

N = length(data);
M = 7;
R = [0 1; -1 0];
result = zeros(N,M);
torsoIdx = [6 7];

for imgnum = 1:N

    points = data(imgnum).annorect.annopoints.point;
    
    torsoPoint1 = getAnnopointById(points,torsoIdx(1));
    torsoPoint2 = getAnnopointById(points,torsoIdx(2));
    if (isempty(torsoPoint1) || isempty(torsoPoint2))
        fprintf('WARNING!\n');
        fprintf('Torso is not annotated in image %d\n', imgnum);
        continue;
    end
    rootPos = [mean([torsoPoint1.x torsoPoint2.x]) mean([torsoPoint1.y torsoPoint2.y])];
        
    annoPoints = zeros(2,2);
    pidx = 1;
    bMissing = false;
    for j=joint.parent.xaxis;
        point = getAnnopointById(points,j);
        if (isempty(point))
            bMissing = true;
            break;
        end
        annoPoints(pidx,1) = point.x;
        annoPoints(pidx,2) = point.y;
        pidx = pidx + 1;
    end
    
    if bMissing
        fprintf('WARNING!\n');
        fprintf('Annopoint %d is missing in image %d\n', j, imgnum);
        continue;
    end
    
    xaxis_parent = (annoPoints(2,:) - annoPoints(1,:));
    xaxis_parent = xaxis_parent/norm(annoPoints(2,:) - annoPoints(1,:));
    xaxis_parent = R * xaxis_parent';
    
    annoPoints = zeros(2,2);
    pidx = 1;
    for j=joint.child.xaxis;
        point = getAnnopointById(points,j);
        if (isempty(point))
            bMissing = true;
            break;
        end
        annoPoints(pidx,1) = point.x;
        annoPoints(pidx,2) = point.y;
        pidx = pidx + 1;
    end
    
    if bMissing
        fprintf('WARNING!\n');
        fprintf('Annopoint %d is missing in image %d\n', j, imgnum);
        continue;
    end
    
    xaxis_child = (annoPoints(2,:) - annoPoints(1,:));
    xaxis_child = xaxis_child/norm(annoPoints(2,:) - annoPoints(1,:));
    xaxis_child = R * xaxis_child';
    
    rot_parent = atan2(xaxis_parent(2), xaxis_parent(1));
    rot_child = atan2(xaxis_child(2), xaxis_child(1));
    
    if (isnan(rot_parent))
        fprintf('WARNING!\n');
        fprintf('The rot_parent is NAN for jidx %d in image %d\n', joint.id, imgnum);
        rot_parent = 0.0;
    end
    
    if (isnan(rot_child))
        fprintf('WARNING!\n');
        fprintf('The rot_child is NAN for jidx %d in image %d\n', joint.id, imgnum);
        rot_child = 0.0;
    end
    
    anglesDiff = rot_child - rot_parent;
    if (isnan(anglesDiff))
        fprintf('WARNING!\n');
        fprintf('The angle is NAN for jidx %d in image %d\n', joint.id, imgnum);
        anglesDiff = 0.0;
    end
    
    rot_child = wrapMinusPiPi(rot_child);
    rot_parent = wrapMinusPiPi(rot_parent);
    anglesDiff = wrapMinusPiPi(anglesDiff);
        
    annoPoints = zeros(2,2);
    pidx = 1;
    for j=joint.child.pos;
        point = getAnnopointById(points,j);
        annoPoints(pidx,1) = point.x;
        annoPoints(pidx,2) = point.y;
        pidx = pidx + 1;
    end
        
    child_pos = [mean([annoPoints(1,1) annoPoints(2,1)]) mean([annoPoints(1,2) annoPoints(2,2)])];
            
    annoPoints = zeros(2,2);
    pidx = 1;
    for j=joint.parent.pos;
        point = getAnnopointById(points,j);
        annoPoints(pidx,1) = point.x;
        annoPoints(pidx,2) = point.y;
        pidx = pidx + 1;
    end
        
    parent_pos = [mean([annoPoints(1,1) annoPoints(2,1)]) mean([annoPoints(1,2) annoPoints(2,2)])];
   
    child_pos = child_pos - rootPos;
    parent_pos = parent_pos - rootPos;
    result(imgnum,:) = [rot_child rot_parent child_pos(1) child_pos(2) parent_pos(1) parent_pos(2) anglesDiff];
end

end