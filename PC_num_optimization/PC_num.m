close all; clear all; clc;

%% Load Data

load('benchmark_displacements') %smaller file to load without all the extra info - this is what is uploaded to github under the pc num opt folder
%{
load('benchmark20mm1000samples.mat');
headneck=xI; clearvars -except headneck;
load('benchmark_aorta.mat')
aorta=xI; clearvars -except headneck aorta;
load('benchmark_kidney')
kidney=xI; clearvars -except headneck aorta kidney;
%}

%% Running Cross Validation for Each Model

PChn=27; 
PCaorta=11;
PCkidney=15;

for i=1:5
    [hn_avgMaxMax(i),hn_avgMeanMax(i)] = crossVal(headneck,PChn+i-3);
    [aorta_avgMaxMax(i),aorta_avgMeanMax(i)] = crossVal(aorta,PCaorta+i-3);
    [kidney_avgMaxMax(i),kidney_avgMeanMax(i)] = crossVal(kidney,PCkidney+i-3);
end

figure(1) %creating plot to show how error changes around the 
subplot(3,1,1)
plot(PChn-2:PChn+2,hn_avgMaxMax,'*-r'); grid on
title('Head & Neck'); xlabel('# of PCs'); ylabel('Mean Max Nodal Displacement Error (mm)')
subplot(3,1,2)
plot(PCaorta-2:PCaorta+2,aorta_avgMaxMax,'*-g'); grid on
title('Aorta'); xlabel('# of PCs'); ylabel('Mean Max Nodal Displacement Error (mm)')
subplot(3,1,3)
plot(PCkidney-2:PCkidney+2,kidney_avgMaxMax,'*-b'); grid on
title('Kidney'); xlabel('# of PCs'); ylabel('Mean Max Nodal Displacement Error (mm)')

%% k-fold Cross Validation
function [avgMaxMax,avgMeanMax] = crossVal(xI,PCnum)

k=5; %k-folds

avg=0;max=0; %initializing variables

    for i=1:k
        test=xI(:,200*(i-1)+1:200*i); %taking out test data
        train=xI; %creating this every loop because in the line below we delete entries from the matrix
        train(:,200*(i-1)+1:200*i)=[]; %deleting test data from training data
        [meanMaxErr,maxMaxErr]=PCA(train,test,PCnum); %calling PCA subfunc
        avg=avg+meanMaxErr; %adding up mean errors
        max=max+maxMaxErr; %adding up max errors
    end
    
    avgMeanMax=avg/k; %taking avg of the sum of errors
    avgMaxMax=max/k;
    
end

function [meanMaxErr,maxMaxErr]=PCA(xI,test,PCnum)
%% Mean Shifting the Data
xI_unshifted=xI; test_unshifted=test;
meanshift=mean(xI,2); testshift=mean(test,2);
xI=xI-meanshift; test=test-testshift;

covArray=xI*xI'; %3n by 3n
[v,d]=eig(covArray);

[eigValFull,order]=sort(diag(d),'descend'); %sorting eigenvalues by size
eigVectFull=v(:,order); %sorting PCs by size

eigVal=eigValFull(1:PCnum); %PCx1
eigVect=eigVectFull(:,1:PCnum); %3nxPC

weights=eigVect'*test; %PCx3n * 3nx200

%% Reconstruction

data_temp=eigVect*weights; %3nx200
data_recon=data_temp+testshift;

%% Error Calculation

rawError=test_unshifted-data_recon;
euclidError=zeros(length(rawError)/3,length(weights)); %nx200

for i=1:length(rawError)/3
    euclidError(i,:)=sqrt(rawError(3*i-2,:).^2+rawError(3*i-1,:).^2+rawError(3*i,:).^2); %3D euclidean error for each node
end

maxErr=max(euclidError)*1000;
maxMaxErr=max(maxErr);
meanMaxErr=mean(maxErr);
end