function [point, ind] = getAnnopointById(points, id)
point = [];
for i=1:length(points)
    if (points(i).id == id)
        point = points(i);
        ind = i;
        return;
    end
end
% assert(0);
end