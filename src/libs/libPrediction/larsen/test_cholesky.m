clear; close all; clc;

%% TEST
% Compare up- and downdate algorithms for the Cholesky factorization with
% the corresponding Cholesky decompositions

%% Updates
X1 = rand(6,4);
x1 = rand(6,1);
R1_0 = chol(X1'*X1);
R1_1 = cholinsert(R1_0, x1, X1);
R1_true = chol([X1 x1]'*[X1 x1]);
assert(norm(R1_1 - R1_true) < 1e-12)

%% Downdates
X2 = rand(4,6);
x2 = rand(4,1);
delta = 1;
R2_0 = chol((X2'*X2 + delta*eye(6)));
R2_1 = cholinsert(R2_0, x2, X2, delta);
R2_true = chol([X2 x2]'*[X2 x2] + delta*eye(7));
assert(norm(R2_1 - R2_true) < 1e-12)
