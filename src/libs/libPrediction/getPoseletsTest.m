function res = getPoseletsTest(poseletResponcesDir, idx_psl, imgidx, root_pos)

if (isempty(idx_psl))
    filename = [poseletResponcesDir '/resp_' padZeros(num2str(imgidx),4) '.mat'];
    r = load(filename);
    resp = r.resp;
    idx_psl = 11:size(resp,1);
end

min_score = 0;

fprintf('image %d\n',imgidx);
filename = [poseletResponcesDir '/resp_' padZeros(num2str(imgidx),4) '.mat'];
r = load(filename);
resp = r.resp;

resp_pos = resp(idx_psl,[3 5 6]);
assert(~isempty(root_pos));
resp_pos(:,2:3) = resp_pos(:,2:3) - repmat([root_pos(:,2) root_pos(:,1)],size(resp_pos,1),1);

resp_pos(find(resp_pos(:,1) < min_score),:) = 0;

resp_pos = reshape(resp_pos,size(resp_pos,1)*size(resp_pos,2),1);

res = resp_pos;

end