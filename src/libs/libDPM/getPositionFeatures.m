function features = getPositionFeatures(annolist,parts,pidx)
%% compute features for clustering similar to Desai&Ramanan, ECCV'12
fprintf('getPositionFeatures()\n');

dist = zeros(length(annolist),2*length(parts));
w = zeros(length(annolist),length(parts));
features = zeros(length(annolist),2*length(parts));
T = ones(1,length(parts))*1e-1;

for imgidx = 1:length(annolist)
%     fprintf('imgidx %d\n',imgidx);
    fprintf('.');
    points = annolist(imgidx).annorect.annopoints.point;
    pp = zeros(length(parts),2);
    for i = 1:length(parts)
        p1 = getAnnopointById(points, parts(i).pos(1));
        p2 = getAnnopointById(points, parts(i).pos(2));
        if (isempty(p1) || isempty(p2))
            pp(i,:) = nan;
        else
            pp(i,:) = [(p1.x + p2.x)/2 (p1.y + p2.y)/2];
        end
    end
    
    for i=1:length(parts)
        idxs = 2*(i-1)+1:2*i;
        dist(imgidx,idxs) = pp(pidx+1,:) - pp(i,:);
        w(imgidx,i) = exp(-T(i)*norm(dist(imgidx,idxs)));
        features(imgidx,idxs) = dist(imgidx,idxs)*w(imgidx,i);
    end
    
    %     for i=1:length(parts)
    %         fprintf('%d: [%f %f]; %f %f\n',i,features(imgidx,2*(i-1)+1:2*i), norm(features(imgidx,2*(i-1)+1:2*i)), norm(dist(imgidx,2*(i-1)+1:2*i)));
    %     end
    if (~mod(imgidx, 100))
        fprintf(' %d/%d\n',imgidx,length(annolist));
    end
end
fprintf('\ndone!\n');
end