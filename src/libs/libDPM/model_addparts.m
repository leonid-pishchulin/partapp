function model = model_addparts(model, lhs, ruleind, filterind, numparts, psize)
% Add part filters to a model.
%
% model      object model
% lhs        add parts to: model.rules{lhs}(ruleind)
% ruleind    add parts to: model.rules{lhs}(ruleind)
% filterind  source filter to initialize parts from
% numparts   number of parts to add
% psize      size of each part

% if the filter is mirrored, find its partner so mirrored
% parts can be added to it as well
if (numparts == 0)
    fprintf('Warning: numparts == 0. Add no parts.\n');
    return;
end

partner = [];
sym = 'N';
if model.filters(filterind).symmetric == 'M'
  bl = model.filters(filterind).blocklabel;
  for i = 1:model.numfilters
    if i ~= filterind && model.filters(i).blocklabel == bl
      partner = i;
      sym = 'M';
      break;
    end
  end
end

source = model.filters(filterind).w;
%%%%%%%%%% Leonid %%%%%%%%%%%%%%%%%%%%%%%%%%%%
source_size = [size(source,1) size(source,2)];
psize_old = psize;
if (psize(1) > 2*source_size(1))
    fprintf('Warning! Changed part size!\n')
    psize(1) = 2*source_size(1);
    fprintf('Old psize(1): %d; new psize(1): %d\n',psize_old(1), psize(1))
end

if (psize(2) > 2*source_size(2))
    fprintf('Warning! Changed part size!\n')
    psize(2) = 2*source_size(2);
    fprintf('Old psize(2): %d; new psize(2): %d\n',psize_old(2), psize(2))
end
%%%%%%%%%% Leonid %%%%%%%%%%%%%%%%%%%%%%%%%%%%

pfilters = mkpartfilters(source, psize, numparts);

for i = 1:numparts
  [model, symbolf, fi] = model_addfilter(model, pfilters(i).w, sym);
  [model, N1] = model_addnonterminal(model);

  % add deformation rule
  defoffset = 0;
  defparams = pfilters(i).alpha*[0.1 0 0.1 0];
  [model, offsetbl, defbl] = model_addrule(model, 'D', N1, symbolf, ...
                                           defoffset, defparams, sym);

  % add deformation symbols to rhs of rule
  anchor1 = pfilters(i).anchor;
  model.rules{lhs}(ruleind).rhs = [model.rules{lhs}(ruleind).rhs N1];
  model.rules{lhs}(ruleind).anchor = [model.rules{lhs}(ruleind).anchor anchor1];

  if ~isempty(partner)
    [model, symbolfp, fi] = model_addmirroredfilter(model, fi);
    [model, N2] = model_addnonterminal(model);

    % add mirrored deformation rule
    model = model_addrule(model, 'D', N2, symbolfp, ...
                          defoffset, defparams, sym, offsetbl, defbl);

    x = pfilters(i).anchor(1) + 1;
    y = pfilters(i).anchor(2) + 1;
    % add deformation symbols to rhs of rule
    x2 = 2*size(source, 2)-(x+psize(2)-1)+1;
    anchor2 = [x2-1 y-1 1];
    model.rules{lhs}(partner).rhs = [model.rules{lhs}(partner).rhs N2];
    model.rules{lhs}(partner).anchor = [model.rules{lhs}(partner).anchor anchor2];
  end
end
