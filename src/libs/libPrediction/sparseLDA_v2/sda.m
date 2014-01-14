function [b, theta, rss] = sda(X,Y,lambda,stop,maxIte,trace)
% [b theta rss] = sda(X,Y,lambda,stop,maxIte,trace) 
% Performs Sparse Linear Disciminant Analysis
% Solving: argmin{|(Y*theta-X*b)|_2^2 + t*|beta|_1 + lambda*|beta|_2^2}
%
% INPUT:
% X      : matrix of n observations down the rows and p variable columns. The
%          columns are assumed normalized
% Y      : matrix initializing the dummy variables representing the groups
% lambda : the weight on the L2-norm for elastic net regression. Default =
%          1e-6
% stop   : nonzero STOP will perform
%          elastic net regression with early stopping. If STOP is negative, its 
%          absolute value corresponds to the desired number of variables. If STOP
%          is positive, it corresponds to an upper bound on the L1-norm of the
%          b coefficients. There is a one to one correspondance between stop
%          and t. Defult = 0.
% maxIte : Maximum number of iterations. Default = 100.
% trace  : If trace is true the display level is on. It is off when trace is false. Default = 0. 
%
% OUTPUT:
% b     : The regression parameters
% theta : Optimal scores
% rss   : Residual Sum of Squares at each itearation
%
% Copyright (c) October, 2007: Line H. Clemmensen, IMM, DTU, lhc@imm.dtu.dk
% Reference: Clemmensen, L., Hastie, T. and Ersbøll, K. (2007), Sparse
% Discriminant Analysis, Technical Report, Informatics and Mathematical
% Modelling, Technical University of Denmark.
% Builds on the elastic net package by Karl Sjôstrand:
% http://www2.imm.dtu.dk/~kas/software/spca/index.html
%

if nargin < 6
    trace = 0;
end
if nargin < 5
    maxIte = 100;
end
if nargin < 4
    stop = 0;
end
if nargin < 3
    lambda = 1e-6;
end

[N,p] = size(X);
K = size(Y,2);
RSS_old = 1e8;
RSS = 1e6;
ite = 0;
Ytheta = Y;
theta = eye(K);
Dpi = Y'*Y/(N); % diagonal matrix of class priors
Dpi_inv = diag(1./sqrt(diag(Dpi)));
if length(stop)<K
    stop = repmat(stop(1),1,K);
end
stop2=stop-2;
for i=1:K
    if stop2(i)<-p
        stop2(i)=-p;
    end
end
rss = zeros(maxIte,1);
b = zeros(p,K);

while (RSS_old-RSS)/RSS > 1e-4 & ite < maxIte 
    RSS_old = RSS;
    ite = ite + 1;
    % 1. Estimate beta:    
    for j=1:K
        [Yc, my] = center(Ytheta(:,j));
        B = larsen(X,Yc,lambda,stop2(j));
        I = find(sum(B~=0,2)==-stop(j));,
        b(:,j) = B(I(end),:)';
        Yhat(:,j) = X*b(:,j)+my;
    end 
    if trace
    RSS = norm(Ytheta-Yhat,'fro')^2;
    disp(sprintf('ite, beta : %d,\t RSS: %1.4f',ite, RSS))
    end
    % 2. Optimal scores: (balanced Procrustes problem)
    B = Y'*Yhat;
    [U,S,V] = svd(B);
    theta_old = theta;
    theta = Dpi_inv*U*V';
    Ytheta = Y*theta;
    RSS = norm(Ytheta-Yhat,'fro')^2;
    rss(ite) = RSS;
    if trace
    disp(sprintf('ite, theta: %d,\t RSS: %1.4f',ite, RSS))
    end
end
if (RSS_old-RSS)<0
   theta = theta_old;
   Ytheta = Y*theta;
   ite = ite - 1;
end
for j=1:K
        [Yc, my] = center(Ytheta(:,j));
        B = larsen(X,Yc,lambda,stop2(j));
        I = find(sum(B~=0,2)==-stop(j));,
        b(:,j) = B(I(end),:)';
        Yhat(:,j) = X*b(:,j)+my;
end    
if trace
RSS = norm(Ytheta-Yhat,'fro')^2;
disp(sprintf('final update: %d,\t RSS: %1.4f',ite, RSS))
end
rss = rss(1:ite);