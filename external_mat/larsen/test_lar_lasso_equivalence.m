clear; close all; clc;

%% TEST
% Assert that LAR and LASSO are equal in cases where no variables are
% dropped

X = gallery('orthog',100,5);
X = X(:,2:6);
y = center(rand(100,1));

b_lar = lar(X,y);
b_lasso = lasso(X, y);

assert(norm(b_lar - b_lasso) < 1e-12)
