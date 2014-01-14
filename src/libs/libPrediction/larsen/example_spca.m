clear; close all; clc;

% Fix stream of random numbers
s1 = RandStream.create('mrg32k3a','Seed', 11);
s0 = RandStream.setDefaultStream(s1);

% Create synthetic data set
n = 1500; p = 500;
t = linspace(0, 1, p);
pc1 = max(0, (t - 0.5)> 0);
pc2 = 0.8*exp(-(t - 0.5).^2/5e-3);
pc3 = 0.4*exp(-(t - 0.15).^2/1e-3) + 0.4*exp(-(t - 0.85).^2/1e-3);
X = [ones(n/3,1)*pc1 + randn(n/3,p); ones(n/3,1)*pc2 + randn(n/3,p);...
  ones(n/3,1)*pc3 + randn(n/3,p)];

% PCA and SPCA
[U D V] = svd(X, 'econ');
d = sqrt(diag(D).^2/n);
K = 3;
delta = inf;
stop = -[250 125 100];
maxiter = 3000;
convergenceCriterion = 1e-9;
verbose = true;

[SL SD] = spca(X, [], K, delta, stop, maxiter, convergenceCriterion, verbose);

figure(1)
plot(t, [pc1; pc2; pc3]); axis([0 1 -1.2 1.2]);
title('Noiseless data');
figure(2);
plot(t, X);  axis([0 1 -6 6]);
title('Data + noise');
figure(3);
plot(t, d(1:3)*ones(1,p).*(V(:,1:3)'));  axis([0 1 -1.2 1.2]);
title('PCA');
figure(4)
plot(t, sqrt(SD)*ones(1,p).*(SL'));  axis([0 1 -1.2 1.2]);
title('SPCA');

% Restore random stream
RandStream.setDefaultStream(s0);
