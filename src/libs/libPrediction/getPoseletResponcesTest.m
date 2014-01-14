function [poseletResponces] = getPoseletResponcesTest(subsetDims, poseletResponcesDir, torsoPosDir, torsoPartIdx, imgidx)

poseletResponces = [];

fprintf('Loading torso position\n');
if ~isempty(torsoPosDir)
    filename = [torsoPosDir '/pose_est_imgidx' padZeros(num2str(imgidx),4) '.mat'];
    pos = load(filename);
    torsoPos = pos.best_conf(torsoPartIdx+1,5:6);
else
    fprintf('WARNING: torsoPosDir is empty!\n');
end

if (~isempty(poseletResponcesDir))
    poseletResponces = getPoseletsTest(poseletResponcesDir, subsetDims, imgidx, torsoPos);
    fprintf('poseletResponces dim: %d\n', size(poseletResponces,2));
else
    fprintf('WARNING: poseletResponcesDir is empty!\n');
end

end