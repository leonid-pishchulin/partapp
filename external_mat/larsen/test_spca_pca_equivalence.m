clear; close all; clc;

%% TEST
% Assert that the full SPCA model and PCA are equal

n = 100;
p = 25;
Z = rand(p);
C = Z'*Z;

X = center(mvnrnd(zeros(1,p), C, n));

K = p; % all possible components
delta = 5; % any value will do
stop = 0; % no L1 constraint
B = spca(X, [], K, delta, stop);

[U D V] = svd(X, 'econ');

assert(norm(abs(V) - abs(B)) < 1e-12)
