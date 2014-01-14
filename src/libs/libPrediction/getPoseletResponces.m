function [poseletResponces, poseParams] = getPoseletResponces(data, subsetDims, joints, poseletResponcesDir, torsoPosDir, torsoPartIdx, nEx, imgidx)

poseletResponces = [];

torsoPos = zeros(nEx,2);

fprintf('Loading torso positions...\n');
if ~isempty(torsoPosDir)
    if (nEx == 1)
        filename = [torsoPosDir '/pose_est_imgidx' padZeros(num2str(imgidx-1),4) '.mat'];
        pos = load(filename);
        torsoPos = pos.best_conf(torsoPartIdx+1,5:6);
    else
        for imgidx = 1:nEx
            filename = [torsoPosDir '/pose_est_imgidx' padZeros(num2str(imgidx-1),4) '.mat'];
            pos = load(filename);
            torsoPos(imgidx,:) = pos.best_conf(torsoPartIdx+1,5:6);
        end
    end
else
    fprintf('WARNING: torsoPosDir is empty!\n');
end

if (~isempty(poseletResponcesDir))
    poseletResponces = getPoselets(poseletResponcesDir, subsetDims, nEx, torsoPos);
    fprintf('poseletResponces dim: %d\n', size(poseletResponces,2));
else
    fprintf('WARNING: poseletResponcesDir is empty!\n');
end

fprintf('\nGet component label features\n');
nJoints = length(joints);
poseParams = cell(nJoints,1);
for j = 1:nJoints
    fprintf('jidx %d\n', j-1);
    poseParams{j} = getPoseParams(data, joints(j));
end

end