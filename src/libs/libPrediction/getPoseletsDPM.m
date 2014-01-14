function fvec = getPoseletsDPM(poseletResponcesDir, idx_psl, nImg, root_pos)

if (isempty(idx_psl))
%     idx_psl = [11 12 14 15 18 19 21];
%     idx_psl = [11:16 18:21];
end

assert(size(root_pos,1) == nImg);

fvec = [];

filename = [poseletResponcesDir '/respAll.mat'];
r = load(filename);
respAllImg = r.resp;
min_score = -3;

for imgidx = 1:nImg
    fprintf('image %d/%d\n',imgidx,nImg);
    resp = respAllImg(imgidx,:);
    resp = reshape(resp', size(resp,2)/3, 3);
    assert(~isempty(root_pos(imgidx,:)));
    resp(:,2:3) = resp(:,2:3) - repmat([root_pos(imgidx,1) root_pos(imgidx,2)],size(resp,1),1);
%     resp(find(resp(:,1) <= min_score),2:3) = 0;
    resp_all = resp;
%     %% shift and normalization
%     min_score = min(resp_all(:,1));
%     resp_all(:,1) = resp_all(:,1) - repmat(min_score, size(resp_all,1), 1);
%     max_score = max(resp_all(:,1));
%     resp_all(:,1) = resp_all(:,1) ./ max_score;
%     %%

    resp_all = resp_all(:,1);
    
    resp_all = reshape(resp_all,size(resp_all,1)*size(resp_all,2),1)';
    fvec(imgidx, :) = resp_all;
end
end