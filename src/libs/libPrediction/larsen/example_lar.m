clear; close all; clc;

% Fix stream of random numbers
s1 = RandStream.create('mrg32k3a','Seed', 22);
s0 = RandStream.setDefaultStream(s1);

% Create data set
n = 100; p = 6;
correlation = 0.6;
Sigma = correlation*ones(p) + (1 - correlation)*eye(p);
mu = zeros(p,1);
X = mvnrnd(mu, Sigma, n);
% Model is lin.comb. of first three variables plus noise
y = X(:,1) + X(:,2) + X(:,3) + 2*randn(n,1);

% Preprocess data
X = normalize(X);
y = center(y);

% Run LAR
[beta info] = lar(X, y, 0, true, true);

% Find best fitting model
[bestAIC bestIdx] = min(info.AIC);
best_s = info.s(bestIdx);

% Plot results
h1 = figure(1);
plot(info.s, beta, '.-');
xlabel('s'), ylabel('\beta', 'Rotation', 0)
line([best_s best_s], [-6 14], 'LineStyle', ':', 'Color', [1 0 0]);
legend('1','2','3','4','5','6',2);

% Restore random stream
RandStream.setDefaultStream(s0);
