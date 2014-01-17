% Example of performing SMDA
% Copyright (c), March 2009, Line H. Clemmensen, DTU Informtics, lhc@imm.dtu.dk

clear all, close all
% Remember to add a path to the elastic net library made by Karl Sjöstrand
% Can be downloaded from:
% http://www2.imm.dtu.dk/~kas/software/spca/index.html
%

% load data: X, and subclass probabilities: Z
load penicilliumYES
[n,p] = size(X);

% construct the Y classes
% 12: Mel, 12: Pol, 12: Ven. (4 isolates x 3 rep.)
Yclass = [zeros(12,1); ones(12,1);2*ones(12,1)];
Rj = [4,4,4]; % the number of subclasses in each class

% the test samples
Io = [3,6,9,12];
Itst = [Io,Io+12,Io+24];

% the training samples:
Itr = setdiff(1:36,Itst);
Xtr = X(Itr,:);
Ztr = Z(Itr,:);
% the number of classes
K = 3;

% set parameters
lambda = 1e-2; % l2-norm
stop = [-5]; % l1-norm. negative: number of vars in LARS-EN
maxiter = 30; % max iter in SDCA alg.

% normalize data
[Xtr,mx,vx] = normalize(Xtr);
In = find(vx~=0);
Xtr = Xtr(:,In);
X = (X(:,In)-ones(n,1)*mx(In))./sqrt(ones(n,1)*vx(In));

% perform SMDA
[sl, theta, Znew, mu, Cov, Dp] = smda(Xtr, Ztr, Rj, lambda, stop, 11, maxiter,1e-6,1);

% Project data onto the sparse directions (dim=11)
['size of beta: ' num2str(size(sl))]
DC = X*sl;

% calcualte class probabilities for the training samples
Rz = [0,Rj];
for ii=1:K
    pr(:,ii) = sum(Znew(:,(sum(Rz(1:ii))+1):(sum(Rz(1:ii))+Rz(ii+1))),2);
end
[G, Yhat] = max(pr,[],2);
% training error
Training_error = sum(Yhat~=(Yclass(Itr)+1))/length(Itr)

% calcualte class probabilities for the test samples
for ii=1:K
    IK = sum(Rj(1:ii-1))+1:sum(Rj(1:ii-1))+Rj(ii);
    Dmahal_K=zeros(n, Rj(ii));
    for jj=1:Rj(ii)
        Dmahal_K(:,jj) = diag((DC-ones(size(DC,1),1)*mu(:,IK(jj),ii)')*(Cov\(DC-...
            ones(size(DC,1),1)*mu(:,IK(jj),ii)')'));
    end
    sum_K = sum(ones(size(DC,1),1)*Dp(IK).*exp(-Dmahal_K/2),2);
    for jj=1:Rj(ii)
        Zt(:,IK(jj)) = Dp(IK(jj))*exp(-Dmahal_K(:,jj)/2)./(sum_K+1e-3);
    end
end
clear pr
for ii=1:K
    pr(:,ii) = sum(Zt(:,(sum(Rz(1:ii))+1):(sum(Rz(1:ii))+Rz(ii+1))),2);
end
[G, Yhat] = max(pr,[],2);
% test error
Test_error = sum(Yhat(Itst)~=(Yclass(Itst)+1))/length(Itst)




% make plots of the sparse directions and the three classes
sm = 100;
I0=1:12;
I1=13:24;
I2=25:36;

figure;
scatter3(DC(I0,1),DC(I0,2),DC(I0,3),'ro','filled','SizeData',sm), hold on
scatter3(DC(I1,1),DC(I1,2),DC(I1,3),'ks','filled','SizeData',sm)
scatter3(DC(I2,1),DC(I2,2),DC(I2,3),'bv','filled','SizeData',sm)
xlabel('1st SD'), ylabel('2nd SD'), zlabel('3rd SD')
legend('Mel','Pol','Ven')

figure;
scatter3(DC(I0,4),DC(I0,5),DC(I0,6),'ro','filled','SizeData',sm), hold on
scatter3(DC(I1,4),DC(I1,5),DC(I1,6),'ks','filled','SizeData',sm)
scatter3(DC(I2,4),DC(I2,5),DC(I2,6),'bv','filled','SizeData',sm)
xlabel('4th SD'), ylabel('5th SD'), zlabel('6th SD')
legend('Mel','Pol','Ven')

figure;
scatter3(DC(I0,7),DC(I0,8),DC(I0,9),'ro','filled','SizeData',sm), hold on
scatter3(DC(I1,7),DC(I1,8),DC(I1,9),'ks','filled','SizeData',sm)
scatter3(DC(I2,7),DC(I2,8),DC(I2,9),'bv','filled','SizeData',sm)
xlabel('7th SD'), ylabel('8th SD'), zlabel('9th SD')
legend('Mel','Pol','Ven')

figure;
scatter(DC(I0,10),DC(I0,11),'ro','filled','SizeData',sm), hold on
scatter(DC(I1,10),DC(I1,11),'ks','filled','SizeData',sm)
scatter(DC(I2,10),DC(I2,11),'bv','filled','SizeData',sm)
xlabel('10th SD'), ylabel('11th SD')
legend('Mel','Pol','Ven')