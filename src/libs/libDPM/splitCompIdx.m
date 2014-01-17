function spos = splitCompIdx(pos, n, compidx)

assert(n == length(unique(compidx)));
assert(size(compidx,1) == size(pos,2))
spos = [];
for i=1:n
  idx = find(compidx==i);
  spos{i} = pos(idx);
end


end