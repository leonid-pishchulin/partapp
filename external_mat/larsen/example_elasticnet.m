clear; close all; clc;

% Fix stream of random numbers
s1 = RandStream.create('mrg32k3a','Seed', 42);
s0 = RandStream.setDefaultStream(s1);

% Create data set
n = 30; p = 40;
correlation = 0.1;
Sigma = correlation*ones(p) + (1 - correlation)*eye(p);
mu = zeros(p,1);
X = mvnrnd(mu, Sigma, n);
% Model is lin.comb. of first three variables plus noise
y = X(:,1) + X(:,2) + X(:,3) + 0.5*randn(n,1);

% Preprocess data
X = normalize(X);
y = center(y);

% Run elastic net
delta = 1e-3;
[beta info] = elasticnet(X, y, delta, 0, true, true);

% Plot results
h1 = figure(1);
plot(info.s, beta, '.-');
xlabel('s'), ylabel('\beta', 'Rotation', 0)

% Restore random stream
RandStream.setDefaultStream(s0);
