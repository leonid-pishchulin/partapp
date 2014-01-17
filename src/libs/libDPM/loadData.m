function [pos, neg, compidx] = loadData(annolistPos,annolistNeg)

fprintf('loadData()\n');

pos = [];
numpos = 0;
compidx = -1*ones(length(annolistPos),1);

% load positive training examples
for imgnum=1:length(annolistPos);
    for ridx = 1:length(annolistPos(imgnum).annorect)
        bbox = zeros(1,4);
        bbox(1) = annolistPos(imgnum).annorect(ridx).x1;
        bbox(2) = annolistPos(imgnum).annorect(ridx).y1;
        bbox(3) = annolistPos(imgnum).annorect(ridx).x2;
        bbox(4) = annolistPos(imgnum).annorect(ridx).y2;

        numpos = numpos + 1;
        compidx(numpos) = annolistPos(imgnum).annorect(1).silhouette.id;
          
        pos(numpos).im = annolistPos(imgnum).image.name;
        pos(numpos).x1 = bbox(1);
        pos(numpos).y1 = bbox(2);
        pos(numpos).x2 = bbox(3);
        pos(numpos).y2 = bbox(4);
        pos(numpos).flip = false;
%         pos(numpos).idx = annolistPos(imgnum).imgidx;
    end
end

fprintf('Num pos: %d\n',numpos);

% load negative training examples
neg = [];
numneg = 0;
for imgnum=1:length(annolistNeg)
    numneg = numneg+1;
    neg(numneg).im = annolistNeg(imgnum).image.name;
    neg(numneg).flip = false;
end
fprintf('Num neg: %d\n',numneg);

end