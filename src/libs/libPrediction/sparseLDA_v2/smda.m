function [b theta Z mu Cov dp] = smda(X, Z, Rj, lambda, stop, Q, maxSteps, tol, verbose)
%SMDA Sparse Discriminant Analysis using a mixture model
%
% BETA = SMDA(X, Z, Rj) performs sparse linear disciminant analysis [1]
% using the predictor variables in X, the class-conditional probabilities
% in Z and the number of subgroups in each of the K groups. Inputs are
% described below in order of declaration.
%
% X        : matrix of n observations down the rows and p variable columns.
%            The columns are assumed normalized
% Z        : matrix initializing the probabilities representing the subgroups
% Rj       : K length vector containing the number of subclasses in each of
%            the K classes
% lambda   : the weight on the L2-norm for elastic net regression. Default =
%            1e-6
% stop     : nonzero STOP will perform
%            elastic net regression with early stopping. If STOP is negative, its
%            absolute value corresponds to the desired number of variables. If STOP
%            is positive, it corresponds to an upper bound on the L1-norm of the
%            b coefficients. There is a one to one correspondance between stop
%            and t. Defult = 0.
% Q        : The number of desired components. Default and maximum is the
%            number of subclasses less one.
% maxSteps : Maximum number of itereations. Default = 100.
% tol      : Tolerance for the stopping criterion (change in RSS). Default is
%            1e-6.
% verbose  : If verbose is true the display level is on. It is off when verbose
%            is false. Default = 0.
%
% OUTPUT:
% b        : The regression parameters
% theta    : Optimal scores
% Z        : Updated subclass probabilities
% mu       : The estimated mean vectors of the projected data Z*b.
% Cov      : The estimated covariance matrix of Z*b.
% Dp       : The estimated class proportions.
%
%
%   Example
%   -------
%   Example introduction here
%
%   Example code here
%
%   References
%   -------
%   [1] L.H Clemmensen .... SDA reference here
%   [2] K. Sj?strand, L.H. Clemmensen. SpaSM, a Matlab Toolbox
%   for Sparse Analysis and Modeling. Journal of Statistical Software
%   x(x):xxx-xxx, 2010.
%
%  See also SLDA, ELASTICNET.

%% Input checking

[n p] = size(X); % n: #observations, p: #variables
K = length(Rj); % number of classes
R = size(Z, 2); % total number of subclasses

if nargin < 9
    verbose = 0;
end
if nargin < 8
    tol = 1e-6;
end
if nargin < 7
    maxSteps = 100;
end
if nargin < 6
    Q=R-1;
elseif Q > R - 1
    Q = R - 1; warning('SpaSM:slda', 'At most R-1 components allowed. Forcing Q = R - 1.')
end
if nargin < 5
    stop = 0;
end
if nargin < 4
    lambda = 1e-6;
end
if nargin < 3
    error('SpaSM:smda', 'Input arguments X, Y and Rj must be specified.');
end



% check stopping criterion
if length(stop) ~= R
    stop = stop(1)*ones(1,R);
end

%% Setup
stepO = 0;
b = zeros(p,Q); % coefficients
[m Isubcl] = max(Z, [], 2);
convergenceCriterion = inf;

dp = sum(Z)/n; % mixing probabilities : sum(Z).^2/n ??
zdp = Z./(ones(n,1)*dp); % class belongings scaled according to priors
theta = zeros(R,Q);

ridgeCostO = inf;
convergenceCriterion = inf;

%% Main loop
while convergenceCriterion > tol && stepO < maxSteps
    stepO = stepO + 1;
    for j = 1:Q % for each discriminative direction
        step = 0; % iteration counter
        ridgeCost = inf;
        convergenceCriterion = inf;
        
        % thetaj are the optimal scores for the jth direction
        thetaj = zeros(R,1);
        thetaj(j) = 1/sqrt(dp(j));
        thetaj = orth_theta(dp, theta, thetaj, R, Q);
        
        while convergenceCriterion > tol && step < maxSteps
            step = step + 1;
            
            % 1. Estimate beta for the jth direction
            Zc = Z*thetaj;
            bj = larsen(X, Zc, lambda, stop(j), [], false, false);
            zhatj = X*bj;
            
            % 2. Estimate theta
            thetaj = orth_theta(dp, theta, zdp'*zhatj, R, Q);
            
            ridgeCost_old = ridgeCost;
            ridgeCost = norm(zhatj - Zc,2)^2 + lambda*norm(bj,2)^2;
            convergenceCriterion = abs(ridgeCost_old - ridgeCost)/ridgeCost;
            if verbose
                fprintf('Step: %d\t\tridge cost: %1.4f\t\t|beta|_1: %0.5g\n', step, ridgeCost, norm(bj,1));
            end
            
            if step == maxSteps
                warning('SpaSM:smda', 'Forced exit. Maximum number of steps reached in slda step.');
            end
            
        end
        
        theta(:,j) = thetaj;
        b(:,j) = bj;
    end
    Zhat = X*b;
    % 3. update parameter estimates for mixture of Gaussians:
    [mu Cov dp Z] = t(dp, Q, R, K, n, Rj, Isubcl, Z, Zhat);
    Ztheta = Z*theta;
    
    ridgeCostO_old = ridgeCostO;
    ridgeCostO = norm(Ztheta - Zhat, 2)^2 + lambda*norm(b, 2)^2;
    
    if verbose
        fprintf('Finally:\tridge cost: %1.4f\t\t|beta|_1: %0.5g\n', ridgeCostO, norm(b,1));
    end
    
    if stepO == maxSteps
        warning('SpaSM:smda', 'Forced exit. Maximum number of steps reached in smda step.');
    end
end

%% Private functions

function [mu Cov Dp, Z] = t(Dp, M, R, K, n, Rj, Isubcl, Z, Zhat)
% this function estimates the means, the pooled within-ckass covariance
% matrix, the class probabilities, and the mixing probabilities for the
% Gaussian mixtures model
Cov = zeros(M, M);
mu = zeros(M, R, K);
for i = 1:K
  IK = (sum(Rj(1:i-1)) + 1):(sum(Rj(1:i-1)) + Rj(i));
  for j = 1:Rj(i)
    Ik = find(Isubcl == IK(j));
    sumZ = sum(Z(Ik, IK(j)));
    mu(:, IK(j), i) = sum((Z(Ik, IK(j))*ones(1, M)).*Zhat(Ik,:), 1)/sumZ;
    Cov = Cov + ((Zhat(Ik, :) - ones(length(Ik), 1)*mu(:, IK(j), i)').*(Z(Ik, IK(j))*ones(1, M)))'...
      *(Zhat(Ik, :) - ones(length(Ik), 1)*mu(:, IK(j), i)')/sumZ; % divide by n ??
  end
end
if condest(Cov) > 1e8
  Cov = Cov + 1e-3*eye(R);
end
for i = 1:K
  IK = (sum(Rj(1: i-1)) + 1):(sum(Rj(1:i-1)) + Rj(i));
  Dmahal_K = zeros(n, Rj(i));
  for j = 1:Rj(i)
    Dmahal_K(:,j) = diag((Zhat - ones(n, 1)*mu(:, IK(j), i)')*(Cov\(Zhat - ...
      ones(n, 1)*mu(:, IK(j), i)')'));
  end
  sum_K = sum(ones(n, 1)*Dp(IK).*exp(-Dmahal_K/2), 2);
  for j = 1:Rj(i)
    Z(:, IK(j)) = Dp(IK(j))*exp(-Dmahal_K(:, j)/2)./(sum_K + 1e-3);
  end
  Dp(IK) = sum(Z(:, IK));
  Dp(IK) = Dp(IK)/sum(Dp(IK));
end

function theta = orth_theta(dp, theta, thetaj, K, Q)
% this procedure adjusts theta to ensure orthogonality of Z*theta
theta_aug = [ones(K,1) theta];
theta_p = theta_aug'.*(ones(Q+1,1)*dp);
thetaj = thetaj - theta_aug*theta_p*thetaj;
theta = thetaj./sqrt(sum(dp'.*thetaj.^2));