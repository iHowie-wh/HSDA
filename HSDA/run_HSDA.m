clear; clc;
addpath('./libsvm-new');
addpath('./liblinear-2.1/matlab');
addpath('./shujuku/4');
addpath('./tool')
addpath('./data')
warning off;
load('berlin_feature.mat');
load('berlin_label.mat');
S = double(feature);
S_Label=double(label);

load('CVE_feature.mat');
load('CVE_label.mat');
T = double(feature(1:350,:));
Ttest= double(feature(351:end,:));
T_Label=double(label(1:350,:));
Ttest_Label = double(label(351:end,:));
%################################################################
S = normalization(S',1);
S =S';
Ttest=normalization(Ttest',1);
Ttest = Ttest';
Ttest = Ttest(:,:);
T = normalization(T',1);
T = T';
Xt = T(:,:);
Xs = S(:,:);
Ys = S_Label;
Yt = T_Label;
test_data = Ttest;
test_label = Ttest_Label;
XS = [Xs;Xt;Ttest];
[COEFF,SCORE, latent] = pca(XS);
SelectNum = cumsum(latent)./sum(latent);
index = find(SelectNum >= 0.98);
pca_dim = index(1);
XS=SCORE(:,1:pca_dim);
Xs = XS(1:size(Xs,1),:);
Xt = XS(size(Xs,1)+1:size(Xs,1)+size(Xt,1),:);
Ttest = XS(size(Xs,1)+size(Xt,1)+1:end,:);
X = [Xs;Xt];%X:n*d
X = X';%X:d*n
Xs = Xs';
Xt = Xt';
obj = [];
loop = 0;
options = [];
cls = [];

options.alpha = 0.1;
options.beta = 0.001;
options.gamma = 10;
options.Niter = 80;
loop = loop + 1;
option = [];
option.ReducedDim = 4;
[P1,~] = PCA1(Xs', option);
[cls,acc,A,obj]=HSDA(Ys,Xs,Xt,options,Ttest,Ttest_Label,P1,loop,obj,cls);
fprintf("------------ Acc:%3.4f ------------\n",acc);
