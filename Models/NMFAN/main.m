%% NMFAN: Regularized nonnegative matrix factorization with adaptive local
%% structure learning
%% Zeng, Xianhua, Shengwei Qu, and Zhilong Wu. "Regularized nonnegative matrix factorization with adaptive local
%% structure learning, Neurocomputing 382 (2020) 196–209.
%% fdsaberi@gmail.com
%% 8/17/2022
clc
clear
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Jaffe Dataset
load('jaffe.mat')
X = NormalizeFea(fea,1);   %% normalize each row of fea to have unit norm;
X = X';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ORL Dataset
% load('ORL.mat')
% X = NormalizeFea(X,1);   %% normalize each row of fea to have unit norm;
% X = X';
% gnd = Y;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Umist Dataset
% load('UMIST.mat')
% X = NormalizeFea(X,1);   %% normalize each row of fea to have unit norm;
% X = X';
% gnd = y;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% warpAR10P Dataset
% load('warpAR10P.mat')
% X = NormalizeFea(X,1);   %% normalize each row of fea to have unit norm;
% X = X';
% gnd = Y;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Yale_32x32 Dataset
% load('Yale_32x32.mat')
% X = NormalizeFea(fea,1);   %% normalize each row of fea to have unit norm;
% X = X';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%YaleB_32x32 Dataset
% load('YaleB_32x32.mat')
% X = NormalizeFea(fea,1);   %% normalize each row of fea to have unit norm;
% X = X';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%CSTR Dataset
% load('CSTR.mat')
% X = NormalizeFea(fea,1);   %% normalize each row of fea to have unit norm;
% X = X';
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Parameters
q = 20; %[10,20,30,40,50,60,70]; %% the pre-defined low-rank parameter
lambda = [0.1,1,100,500,1000];
maxiter = 100; %%  the maximum number of iterations for the method
c1 =  length(unique(gnd));  %% The number of cluters used in kmeans
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%   Constrcution of the affinity matrix S
k = 5;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% NMFAN method
ACCmean  = zeros(length(lambda),1);
NMImean  = zeros(length(lambda),1);
%%
ACCmax  = zeros(length(lambda),1);
NMImax  = zeros(length(lambda),1);
for i=1:length(lambda) %% Grid Search for lambda
    i
    [V, ~] = NMFAN(X, c1, k, lambda(i), maxiter);
    %% Clustering
    tempNMI=zeros(20,1);
    tempACC=zeros(20,1);
    for j = 1:20
        IDX=[];
        IDX = kmeans(V,c1); %%kmeans
        tempNMI(j) = nmi(gnd,IDX);   %%NMI
        tempACC(j) = clusterAccMea(gnd,IDX); %%ACC
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ACCmean(i) = mean(tempACC);
    NMImean(i) = mean(tempNMI);
    %%
    ACCmax(i) = max(tempACC);
    NMImax(i) = max(tempNMI);
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ACCfinalmean=max(ACCmean);
NMIfinalmean=max(NMImean);
%%
ACCfinalmax=max(ACCmax);
NMIfinalmax=max(NMImax);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('The results of running the Kmeans method 20 times and the average of 20 runs')
disp('****************************************')
disp('****************************************')
disp('ACC =')
disp(100*ACCfinalmean)
disp('****************************************')
disp('NMI =')
disp(100*NMIfinalmean)
disp('****************************************')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('The results of running the Kmeans method 20 times and the report of maximum of 20 runs')
disp('****************************************')
disp('****************************************')
disp('ACC - max =')
disp(100*ACCfinalmax)
disp('****************************************')
disp('NMI - max =')
disp(100*NMIfinalmax)
disp('****************************************')