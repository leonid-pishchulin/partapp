function [b theta] = slda(X, Y, lambda2, stop, Q, maxSteps, tol, verbose)
%SLDA Sparse Linear Discriminant Analysis
%
%   BETA = SLDA(X, Y) performs sparse linear disciminant analysis [1] using
%   the predictor variables in X and the dummy encoded class-belongings in
%   Y. X and Y are required inputs. Inputs are described below in order of
%   declaration.
%
%   X         matrix of n observations down the rows and p variable
%             columns. The columns are assumed centered and normalized to
%             Euclidean length 1.
%   Y         matrix initializing the dummy variables representing the
%             classes, e.g Y = [1 0; 1 0; 1 0; 0 1; 0 1] for two classes
%             with the three first observations belonging to class 1 and
%             the last two belonging to class 2.
%   LAMBDA2   the weight on the L2-norm for elastic net regression. Default
%             is LAMBDA2 = 1e-6.
%   STOP      nonzero STOP will perform elastic net regression with early
%             stopping. If STOP is negative, its absolute value corresponds
%             to the desired number of variables. If STOP is positive, it
%             corresponds to an upper bound on the L1-norm of the BETA
%             coefficients. Defult is STOP = -ceil(p/2), corresponding to
%             ceil(p/2) non-zero elements in each discriminative direction.
%   Q         Number of desired discriminative directions. Default value
%             for Q is one less than the number of classes.
%   MAXSTEPS  Maximum number of iterations. Default is MAXSTEPS = 100.
%   TOL       Tolerance for the stopping criterion (change in ridge cost
%             function). Default is TOL = 1e-6.
%   VERBOSE   With VERBOSE set to true, the ridge cost and the L1 norm of
%             the beta coefficients will be printed for each iteration. By
%             default, VERBOSE is set to false.
%
%   OUTPUT:
%   b         The regression parameters. X*b is the data projected onto the
%             discriminative directions.
%   theta     The optimal scores.
%
%   Example
%   -------
%   The example here is of a data set with three classes. The observations
%   corresponding to each class has mean values of 0.6 for 10 out of a
%   total of 150 variables. SLDA is used to derive two discriminative
%   directions, each described by a linear combination of 30 variables. The
%   data projected onto these directions are then used as input to a
%   standard linear classifier and compared to a linear classifier on the
%   raw data.
%
%   % Fix stream of random numbers
%   s1 = RandStream.create('mrg32k3a','Seed', 50);
%   s0 = RandStream.setDefaultStream(s1);
% 
%   p = 150; % number of variables
%   nc = 100; % number of observations per class
%   n = 3*nc; % total number of observations
%   m1 = 0.6*[ones(10,1); zeros(p-10,1)]; % c1 mean
%   m2 = 0.6*[zeros(10,1); ones(10,1); zeros(p-20,1)]; % c2 mean
%   m3 = 0.6*[zeros(20,1); ones(10,1); zeros(p-30,1)]; % c3 mean
%   S = 0.6*ones(p) + 0.4*eye(p); % covariance is 0.6
% 
%   % training data
%   c1 = mvnrnd(m1,S,nc); % class 1 data
%   c2 = mvnrnd(m2,S,nc); % class 2 data
%   c3 = mvnrnd(m3,S,nc); % class 3 data
%   X = [c1; c2; c3]; % training data set
%   Y = [[ones(nc,1);zeros(2*nc,1)] [zeros(nc,1); ones(nc,1); zeros(nc,1)] [zeros(2*nc,1); ones(nc,1)]];
% 
%   % test data
%   c1 = mvnrnd(m1,S,nc);
%   c2 = mvnrnd(m2,S,nc);
%   c3 = mvnrnd(m3,S,nc);
%   X_test = [c1; c2; c3];
% 
%   % SLDA parameters
%   lambda2 = 1e-3; % l2-norm constraint
%   stop = -30; % request 30 non-zero variables
%   maxiter = 250; % maximum number of iterations
%   Q = 2; % request two discriminative directions
%   tol = 1e-6;
% 
%   % normalize training and test data
%   [X mu d] = normalize(X);
%   X_test = (X_test-ones(n,1)*mu)./sqrt(ones(n,1)*d);
% 
%   % run SLDA
%   [beta theta] = slda(X, Y, lambda2, stop, Q, maxiter, tol, true);
% 
%   % Project data onto the sparse directions
%   DC = X*beta;
%   DC_test = X_test*beta;
% 
%   % Classification (LDA of projected data)
%   Yc = [ones(nc,1); 2*ones(nc,1); 3*ones(nc,1)];
%   [class err] = classify(DC, DC, Yc, 'linear');
%   [class_test] = classify(DC_test, DC, Yc, 'linear');
%   err_test = sum(Yc ~= class_test)/length(Yc);
%   fprintf('SLDA result: training error is %2.1f %%, test error is %2.1f %%.\n', 100*err, 100*err_test);
% 
%   [class err] = classify(X, X, Yc, 'linear');
%   [class_test] = classify(X_test, X, Yc, 'linear');
%   err_test = sum(Yc ~= class_test)/length(Yc);
%   fprintf('LDA result: training error is %2.1f %%, test error is %2.1f %%.\n', 100*err, 100*err_test);
%  
%   % plot sparse discriminative directions for test data
%   figure;
%   plot(DC_test(1:nc,1), DC_test(1:nc,2),'ro'), hold on
%   plot(DC_test((nc+1):2*nc,1), DC_test((nc+1):2*nc,2),'ks')
%   plot(DC_test((2*nc+1):3*nc,1), DC_test((2*nc+1):3*nc,2),'bv')
%   xlabel('1st direction'), ylabel('2nd direction')
%   legend('C_1','C_2','C_3','Location','SouthEast')
% 
%   % Restore random stream
%   RandStream.setDefaultStream(s0);
%
%   References
%   -------
%   [1] L.H Clemmensen .... SDA reference here
%   [2] K. Sjöstrand, L.H. Clemmensen, M. Mørup. SpaSM, a Matlab Toolbox
%   for Sparse Analysis and Modeling. Journal of Statistical Software
%   x(x):xxx-xxx, 2010.
%
%  See also SMDA, ELASTICNET.

%% Input checking
if nargin < 2
  error('SpaSM:slda', 'Input arguments X and Y must be specified.');
end

[n p] = size(X); % n: #observations, p: #variables
K = size(Y,2); % K is the number of classes

if nargin < 8
  verbose = false;
end
if nargin < 7
  tol = 1e-6;
end
if nargin < 6
  maxSteps = 100;
end
if nargin < 5
  Q = K - 1; % Q is the number of components
elseif Q > K - 1
  Q = K - 1; warning('SpaSM:slda', 'At most K-1 components allowed. Forcing Q = K - 1.')
end
if nargin < 4
  stop = ceil(p/2);
end
if nargin < 3
  lambda2 = 1e-6;
end

% check stopping criterion
if length(stop) ~= K
  stop = stop(1)*ones(1,K);
end

%% Setup
dp = sum(Y.^2)/n; % diagonal matrix of class priors
ydp = Y./(ones(n,1)*dp); % class belongings scaled according to priors
b = zeros(p,Q); % coefficients of discriminative directions
theta = zeros(K,Q); % optimal scores

%% Main loop
for j = 1:Q
  step = 0; % iteration counter
  ridgeCost = inf;
  convergenceCriterion = inf;
  
  % thetaj are the optimal scores for the jth direction
  thetaj = zeros(K,1);
  thetaj(j) = 1/sqrt(dp(j));
  thetaj = orth_theta(dp, theta, thetaj, K, Q);
  
  while convergenceCriterion > tol && step < maxSteps
    step = step + 1;
    
    % 1. Estimate beta for the jth direction
    Yc = Y*thetaj;
    bj = larsen(X, Yc, lambda2, stop(j), [], false, false);
    yhatj = X*bj;
    
    % 2. Estimate theta
    thetaj = orth_theta(dp, theta, ydp'*yhatj, K, Q);
    
    ridgeCost_old = ridgeCost;
    ridgeCost = norm(yhatj - Yc,2)^2 + lambda2*norm(bj,2)^2;
    convergenceCriterion = abs(ridgeCost_old - ridgeCost)/ridgeCost;
    
    if verbose
      fprintf('Step: %d\t\tridge cost: %1.4f\t\t|beta|_1: %0.5g\n', step, ridgeCost, norm(bj,1));
    end
    
    if step == maxSteps
      warning('SpaSM:slda', 'Forced exit. Maximum number of steps reached.');
    end
    
  end
  
  theta(:,j) = thetaj;
  b(:,j) = bj;
end

if verbose
  Yhat = X*b;
  ridgeCost = norm(Y*theta - Yhat, 2)^2 + lambda2*norm(b, 2)^2;
  fprintf('Finally:\tridge cost: %1.4f\t\t|beta|_1: %0.5g\n', ridgeCost, norm(b,1));
end


%% Private functions
function theta = orth_theta(dp, theta, thetaj, K, Q)
% this procedure adjusts theta to ensure orthogonality of Y*theta
theta_aug = [ones(K,1) theta];
theta_p = theta_aug'.*(ones(Q+1,1)*dp);
thetaj = thetaj - theta_aug*theta_p*thetaj;
theta = thetaj./sqrt(sum(dp'.*thetaj.^2));
