function features = getRotationFeatures(annolist, parts, pidx)

fprintf('getRotationFeatures()\n');

R = [0 1; -1 0];
features = zeros(length(annolist),1);

for imgnum = 1:length(annolist)

    points = annolist(imgnum).annorect.annopoints.point;
    annoPoints = zeros(2,2);
    
    bMissing = false;
    for i = 1:length(parts)
        if (parts(i).id == pidx)
            ii = 1;
            for j = parts(i).xaxis
                 p = getAnnopointById(points, j);
                 if (isempty(p))
                     bMissing = true;
                     break;
                 end
                 annoPoints(ii,1) = p.x;
                 annoPoints(ii,2) = p.y;
                 ii = ii + 1;
            end
            break;
        end
    end
    
    if bMissing
        fprintf('WARNING!\n');
        fprintf('Annopoint %d is missing in image %d\n', j, imgnum);
        continue;
    end
    
    xaxis = (annoPoints(2,:) - annoPoints(1,:));
    xaxis = xaxis/norm(annoPoints(2,:) - annoPoints(1,:));
    xaxis = R * xaxis';
    
    rot = atan2(xaxis(2), xaxis(1));
    
    if (isnan(rot))
        fprintf('WARNING!\n');
        fprintf('rot is NAN for pidx %d in image %d\n', pidx, imgnum);
        rot = 0.0;
    end
    
    features(imgnum) = wrapMinusPiPi(rot);

end


end