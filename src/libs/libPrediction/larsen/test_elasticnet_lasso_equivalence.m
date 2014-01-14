clear; close all; clc;

%% TEST
% Assert that Elastic Net models are equivalent to those obtained by
% running LASSO with Elastic Net-style augmented data matrices

n = 100;
p = 25;

delta = 10;

X = normalize(rand(n, p));
y = center(rand(n,1));

Xtilde = [X; sqrt(delta)*eye(p)];
ytilde = [y; zeros(p,1)];

beta_en = larsen(X, y, delta, 0, [], false, false);
beta_lasso = lasso(Xtilde, ytilde, 0, false, false);
assert(norm(beta_en - beta_lasso) < 1e-12);
