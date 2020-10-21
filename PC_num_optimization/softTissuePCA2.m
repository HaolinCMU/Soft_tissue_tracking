close all; clear all; clc;

%% Load Data
load('benchmark20mm1000samples.mat');

train=xI(:,1:800);
test=xI(:,801:1000);

%% PCA
covArray=cov(train',1); %transposing array - want to see which nodes are important, thus we should expect a 3474x3474 covArray. The "1" flag does the mean shift

[v,d] = eig(covArray); %cant use eigs() bc non square matrix

[sorted_val,order]=sort(diag(d),'descend'); %sorting eigenvalues by size
PC=v(:,order); %sorting PCs by size

explained=sorted_val/sum(sorted_val)*100; % percent variance explained by each each PC

figure(1); %creates plot to show variance explained by PCs, note the diminishing returns around 10 PCs
plot(cumsum(explained(1:50)),'.-');
xlabel('Number of PCs');
ylabel('Cumulative % variance explained');

%% Transforming Data w/ Selected PCs
keep = 1:20;

weights=train'*PC; %Transforming into PC space
transformed=weights(:,keep)*PC(:,keep)'; %Transforming back out of PC Space

transformedBack=transformed+mean(train'); %Adding back the mean

weights_test=test'*PC; %test data
transformed_test=weights_test(:,keep)*PC(:,keep)';

transformedBack_test=transformed_test+mean(train'); %Adding back the mean

%% Error Calculation
rawError=train'-transformedBack;
euclidError=zeros(800,length(rawError)/3);

rawError_test=test'-transformedBack_test;
euclidError_test=zeros(200,length(rawError_test)/3);

for i=1:length(rawError)/3
    euclidError(:,i)=sqrt(rawError(:,3*i-2).^2+rawError(:,3*i-1).^2+rawError(:,3*i).^2); %calculating 3d euclidian distance between actual node deformation and reconstructed deformation
    euclidError_test(:,i)=sqrt(rawError_test(:,3*i-2).^2+rawError_test(:,3*i-1).^2+rawError_test(:,3*i).^2);
end

maxTrain=max(euclidError,[],2)*1000; %converting from m to mm, the '2' flag just specifies direction to take max along
meanTrain=mean(euclidError,2)*1000;
maxTest=max(euclidError_test,[],2)*1000;
meanTest=mean(euclidError_test,2)*1000;

%% Plotting Errors
figure(2)
subplot(2,2,1)
histogram(maxTrain)
title('Max Error (Training Data)')
xlabel('Deformation Error (mm)');ylabel('Frequency')
str={'Max Error:', num2str(max(maxTrain)), 'mm'};
annotation('textbox',[.3 .5 1 .3],'String',str,'FitBoxToText','on');

subplot(2,2,2)
histogram(meanTrain)
title('Mean Error (Training Data)')
xlabel('Deformation Error (mm)');ylabel('Frequency')

subplot(2,2,3)
histogram(maxTest)
title('Max Error (Test Data)')
xlabel('Deformation Error (mm)');ylabel('Frequency')
str={'Max Error:', num2str(max(maxTest)), 'mm'};
annotation('textbox',[.3 .1 1 .3],'String',str,'FitBoxToText','on');

subplot(2,2,4)
histogram(meanTest)
title('Mean Error (Test Data)')
xlabel('Deformation Error (mm)');ylabel('Frequency')



%% Running Again Until Max Error <1mm

keep=1:26; %MODIFY THIS TO CHANGE # of PCs USED
weights=train'*PC; %Transforming into PC space
transformed=weights(:,keep)*PC(:,keep)'; %Transforming back out of PC Space

transformedBack=transformed+mean(train'); %Adding back the mean

weights_test=test'*PC; %test data
transformed_test=weights_test(:,keep)*PC(:,keep)';

transformedBack_test=transformed_test+mean(train'); %Adding back the mean

%% Error Calculation
rawError=train'-transformedBack;
euclidError=zeros(800,length(rawError)/3);

rawError_test=test'-transformedBack_test;
euclidError_test=zeros(200,length(rawError_test)/3);

for i=1:length(rawError)/3
    euclidError(:,i)=sqrt(rawError(:,3*i-2).^2+rawError(:,3*i-1).^2+rawError(:,3*i).^2);
    euclidError_test(:,i)=sqrt(rawError_test(:,3*i-2).^2+rawError_test(:,3*i-1).^2+rawError_test(:,3*i).^2);
end

maxTrain=max(euclidError,[],2)*1000;
meanTrain=mean(euclidError,2)*1000;
maxTest=max(euclidError_test,[],2)*1000;
meanTest=mean(euclidError_test,2)*1000;

%% Plotting Errors
figure(3)
subplot(2,2,1)
histogram(maxTrain)
title('Max Error (Training Data)')
xlabel('Deformation Error (mm)');ylabel('Frequency')
str={'Max Error:', num2str(max(maxTrain)), 'mm'};
annotation('textbox',[.3 .5 1 .3],'String',str,'FitBoxToText','on');

subplot(2,2,2)
histogram(meanTrain)
title('Mean Error (Training Data)')
xlabel('Deformation Error (mm)');ylabel('Frequency')

subplot(2,2,3)
histogram(maxTest)
title('Max Error (Test Data)')
xlabel('Deformation Error (mm)');ylabel('Frequency')
str={'Max Error:', num2str(max(maxTest)), 'mm'};
annotation('textbox',[.3 .1 1 .3],'String',str,'FitBoxToText','on');

subplot(2,2,4)
histogram(meanTest)
title('Mean Error (Test Data)')
xlabel('Deformation Error (mm)');ylabel('Frequency')

%% Saving Variables

meanshift=mean(xI);
PC=PC;
minPC=PC(:,1:26);
weights=[weights;weights_test]';

save('benchmarkPCA.mat','meanshift','PC','minPC','weights')
