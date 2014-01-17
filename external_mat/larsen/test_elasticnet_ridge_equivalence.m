clear; close all; clc;

%% TEST
% Assert that the full Elastic Net model and Ridge Regression are equal

n = 100;
p = 25;

X = normalize(rand(n, p));
y = center(rand(n,1));

beta_en = larsen(X, y, 1e-9, 0, [], false, false);
beta_ridge = (X'*X + 1e-9*eye(p))\X'*y;
assert(norm(beta_en - beta_ridge) < 1e-12);

beta_en = larsen(X, y, 1e-2, 0, [], false, false);
beta_ridge = (X'*X + 1e-2*eye(p))\X'*y;
assert(norm(beta_en - beta_ridge) < 1e-12);

beta_en = larsen(X, y, 1e2, 0, [], false, false);
beta_ridge = (X'*X + 1e2*eye(p))\X'*y;
assert(norm(beta_en - beta_ridge) < 1e-12);

beta_en = larsen(X, y, 1e9, 0, [], false, false);
beta_ridge = (X'*X + 1e9*eye(p))\X'*y;
assert(norm(beta_en - beta_ridge) < 1e-12);
