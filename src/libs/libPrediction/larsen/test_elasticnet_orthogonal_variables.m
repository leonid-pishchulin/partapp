clear; close all; clc;

%% TEST
% Assert that Elastic Net gives expected results when run with orthogonal
% predictors.

X = gallery('orthog',100,5);
X = X(:,2:6);
y = center(1:100)';
delta = 10;

[b_en info] = elasticnet(X, y, delta);

b_ols = X'*y;
b_en2 = zeros(size(b_en));
for i = 1:info.steps
  b_en2(:,i) = sign(b_ols).*max(abs(b_ols) - info.lambda(i)/2,0)/(1 + delta);
end
b_en2 = b_en2*(1 + delta); % to non-naïve solution

assert(norm(b_en - b_en2) < 1e-12)
