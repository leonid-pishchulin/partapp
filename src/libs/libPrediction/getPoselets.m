function res = getPoselets(poseletResponcesDir, idx_psl, nImg, root_pos)

if (isempty(idx_psl))
    filename = [poseletResponcesDir '/resp_' padZeros(num2str(0),4) '.mat'];
    r = load(filename);
    resp = r.resp;
    idx_psl = 11:size(resp,1);
end

assert(size(root_pos,1) == nImg);

nDim = (length(idx_psl))*3;
res = zeros(nImg, nDim);

min_score = 0;

for imgidx = 1:nImg
    fprintf('.');
    filename = [poseletResponcesDir '/resp_' padZeros(num2str(imgidx-1),4) '.mat'];
    r = load(filename);
    resp = r.resp;
    respTop = resp;
    
    resp_pos = resp(idx_psl,[3 5 6]);
    assert(~isempty(root_pos(imgidx,:)));
    resp_pos(:,2:3) = resp_pos(:,2:3) - repmat([root_pos(imgidx,2) root_pos(imgidx,1)],size(resp_pos,1),1);

    resp_pos(find(resp_pos(:,1) < min_score),:) = 0;
    
    resp_pos = reshape(resp_pos,size(resp_pos,1)*size(resp_pos,2),1);
    
    res(imgidx, :) = resp_pos;
    if (~mod(imgidx, 100))
        fprintf(' %d/%d\n',imgidx,nImg);
    end
end
fprintf('\ndone\n');
end