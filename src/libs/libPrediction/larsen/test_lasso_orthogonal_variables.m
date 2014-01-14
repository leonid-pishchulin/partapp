clear; close all; clc;

%% TEST
% Assert that LASSO gives expected results when run with orthogonal
% predictors. 

n = 100;
p = 10;
X = gallery('orthog',n,5);
X = X(:,2:p+1);
y = center((1:100)');

b_lasso = lasso(X, y);
b_ols = X'*y;

% First compare using theoretical value of lambda/2 at each breakpoint.
% Call this value gamma
gamma = [sort(abs(b_ols), 'descend'); 0];
b_lasso2 = zeros(size(b_lasso));
for i = 1:length(gamma)
  b_lasso2(:,i) = sign(b_ols).*max(abs(b_ols) - gamma(i),0);
end

assert(norm(b_lasso - b_lasso2) < 1e-12)

% Then compare for some arbitrary value of lambda, the value of lambda is
% given by the lasso procedure
t = 150; % constraint on the L1 norm of beta
[b_lasso info] = lasso(X, y, t, false);
b_lasso2 = sign(b_ols).*max(abs(b_ols) - info.lambda/2, 0);

assert(norm(b_lasso - b_lasso2) < 1e-12)
