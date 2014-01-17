clear; close all; clc;

% Fix stream of random numbers
s1 = RandStream.create('mrg32k3a','Seed', 50);
s0 = RandStream.setDefaultStream(s1);

p = 150; % number of variables
nc1 = 100; % number of observations per class
nc2 = 100; % number of observations per class
nc3 = 100; % number of observations per class
n = nc1+nc2+nc3; % total number of observations
m1 = 0.6*[ones(10,1); zeros(p-10,1)]; % c1 mean
m2 = 0.6*[zeros(10,1); ones(10,1); zeros(p-20,1)]; % c2 mean
m3 = 0.6*[zeros(20,1); ones(10,1); zeros(p-30,1)]; % c3 mean
S = 0.6*ones(p) + 0.4*eye(p); % covariance is 0.6

% training data
c1 = mvnrnd(m1,S,nc1); % class 1 data
c2 = mvnrnd(m2,S,nc2); % class 2 data
c3 = mvnrnd(m3,S,nc3); % class 3 data
X = [c1; c2; c3]; % training data set
Y = [[ones(nc1,1);zeros(nc2+nc3,1)] [zeros(nc1,1); ones(nc2,1); zeros(nc3,1)] [zeros(nc1+nc2,1); ones(nc3,1)]];

% test data
c1 = mvnrnd(m1,S,nc1);
c2 = mvnrnd(m2,S,nc2);
c3 = mvnrnd(m3,S,nc3);
X_test = [c1; c2; c3];

% SLDA parameters
delta = 1e-3; % l2-norm constraint
stop = -30; % request 30 non-zero variables
maxiter = 250; % maximum number of iterations
Q = 2; % request two discriminative directions
tol = 1e-6;

% normalize training and test data
[X mu d] = normalize(X);
X_test = (X_test-ones(n,1)*mu)./sqrt(ones(n,1)*d);

% run SLDA
[B theta] = slda(X, Y, delta, stop, Q, maxiter, tol, true);

% Project data onto the sparse directions
DC = X*B;
DC_test = X_test*B;

% Classification (LDA of projected data)
Yc = [ones(nc1,1); 2*ones(nc2,1); 3*ones(nc3,1)];
[class err] = classify(DC, DC, Yc, 'linear');
[class_test] = classify(DC_test, DC, Yc, 'linear');
err_test = sum(Yc ~= class_test)/length(Yc);
fprintf('SLDA result: training error is %2.1f %%, test error is %2.1f %%.\n', 100*err, 100*err_test);

[class err] = classify(X, X, Yc, 'linear');
[class_test] = classify(X_test, X, Yc, 'linear');
err_test = sum(Yc ~= class_test)/length(Yc);
fprintf('LDA result: training error is %2.1f %%, test error is %2.1f %%.\n', 100*err, 100*err_test);
 
% plot sparse discriminative directions for test data
figure;
plot(DC_test(1:nc1,1), DC_test(1:nc1,2),'ro'), hold on
plot(DC_test((nc1+1):(nc1+nc2),1), DC_test((nc1+1):(nc1+nc2),2),'ks')
plot(DC_test(((nc1+nc2)+1):(nc1+nc2+nc3),1), DC_test(((nc1+nc2)+1):(nc1+nc2+nc3),2),'bv')
xlabel('1st direction'), ylabel('2nd direction')
legend('C_1','C_2','C_3','Location','SouthEast')

% Restore random stream
RandStream.setDefaultStream(s0);

