% Remember to add a path to the elastic net library made by Karl Sjöstrand
% Can be downloaded from:
% http://www2.imm.dtu.dk/~kas/software/spca/index.html
%

% Example of performing SDA
% Copyright (c), March 2009, Line H. Clemmensen, DTU Informtics, lhc@imm.dtu.dk
clear all, close all
% Remember to add a path to the elastic net library made by Karl Sjöstrand
% Can be downloaded from:
% http://www2.imm.dtu.dk/~kas/software/spca/index.html
%

% load data: X, and dummy variables: Y
load penicilliumYES
[n,p] = size(X);

% construc Yclass indicator
% 12: Mel, 12: Pol, 12: Ven. (4 isolates x 3 rep.)
Yclass = [zeros(12,1); ones(12,1); 2*ones(12,1)];
% class indices
I0=1:12;
I1=13:24;
I2=25:36;

% the test samples
Io = [3,6,9,12];
Iout = [Io,Io+12,Io+24];

% the training samples:
Itr = setdiff(1:36,Iout);
Xtr = X(Itr,:);
Ytr = Y(Itr,:);
K = 3;

% set parameters
lambda = 1e-6; % l2-norm
stop = [-1]; % l1-norm. negative: number of vars in LARS-EN
maxiter = 25; % max iter in SDCA alg.

% normalize data
[Xtr,mx,vx] = normalize(Xtr);
In = find(vx~=0);
Xtr = Xtr(:,In);
X = (X(:,In)-ones(n,1)*mx(In))./sqrt(ones(n,1)*vx(In));

% perform SDA
[sl theta] = slda(Xtr, Ytr, lambda, stop, 2,maxiter, 1e-6,1);
% Project data onto the sparse directions (dim=2)
DC = X*sl;

% Classification (LDA of projected data)
[class,trerr] = classify(DC(Itr,:),DC(Itr,:),Yclass(Itr),'linear');
trerr = length(find(class~=Yclass(Itr)))/length(Yclass(Itr))

[class,tsterr] = classify(DC(Iout,:),DC(Itr,:),Yclass(Itr),'linear');
tsterr = length(find(class~=Yclass(Iout)))/length(Yclass(Iout))

% plot sparse discriminative directions
figure;
plot(DC(I0,1),DC(I0,2),'ro','linewidth',2,'MarkerSize',9), hold on
plot(DC(I1,1),DC(I1,2),'ks','linewidth',2,'MarkerSize',9)
plot(DC(I2,1),DC(I2,2),'bv','linewidth',2,'MarkerSize',9)
xlabel('1st SD'), ylabel('2nd SD')
legend('Mel','Pol','Ven',4)
