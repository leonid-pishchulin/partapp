function [X,mx,varx] = normalize(X)
% NORMALIZE  Normalize the observations of a data matrix.
%    X = NORMALIZE(X) centers and scales the observations of a data
%    matrix such that each variable (column) has unit length.
%
% Author: Karl Skoglund, IMM, DTU, kas@imm.dtu.dk
% Modified: Line Clemmensen, IMM, DTU, lhc@imm.dtu.dk

[n p] = size(X);
[X, mx]= center(X);
varx = sum(X.^2);
X = X./sqrt(ones(n,1)*varx);
