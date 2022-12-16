% ----------------------- # ATFAliTa 2022 --------------------------------%
clc;close all;clear; rng('default') 
% Load Data ---------------------------------------------------------------
load('Train_B0')                                        % train data [N_tr * d], train target [N_tr * 3]
load('Test_B0')                                         % test data  [N_tst * d], test target [N_tr * 3]
N_train=size(Train_B0.Data,1);                          % number of train samples
N_test=size(Test_B0.Data,1);                            % number of test samples
% plot3(Train_B0.target(:,1),Train_B0.target(:,2),Train_B0.target(:,3),'o');grid on;
% Label preparation % Labels are defined with 1 and -1  -------------------
class_number=length(unique(Train_B0.target(:,3)));      % number of classes
train_targ=-ones(N_train,class_number);                 % [N_train * Class]
for i=1:N_train
    train_targ(i,Train_B0.target(i,3))=1;
end

%% Feature Mapping using ELM AE -------------------------------------------
ELMAE_selection=2; % ELMAE_selection=1 >>ELM-AE==(1) else >>sparseELM-AE==(2)
% 1th AE layer ............................................................
bias_coeff=1;
N_ae1=30;                                               % number of neurons of autoencoder layer
lambda_ae1=0.001;                                       % regularization parameter of autoencoder
Wb_ae1=2*rand(size(Train_B0.Data,2)+1,N_ae1)-1;         % random weights and bias generation
% y = (b – a ) * rand() + a   [ a,b ]
X_b1 = [Train_B0.Data bias_coeff * ones(N_train,1)];                              
H1 =logsig(X_b1 * Wb_ae1); % tanh, logsig, ...          % hidden layer output generation g(xw+b)
if ELMAE_selection==1  % (1)
    A = lambda_ae1*eye(size(H1,2),size(H1,2))+(H1'*H1); % (lambda * I + H'H)^-1 *H'*X
    D =  H1'*Train_B0.Data;
    Beta_ae1 =pinv(A)*D;                                % Moore-Penrose pseudoinverse
else        % (2)
    q=1; iter=100;     % SIFTA parameters
    Beta_ae1  =  sparse_elm_autoencoder(H1,Train_B0.Data,q,iter);
end
Xtr_new1= logsig(Train_B0.Data * Beta_ae1');
[Xtr_new1 , PS_k1]= mapminmax(Xtr_new1' , 0 ,1);        % normalization on features
Xtr_new1=Xtr_new1';
% 2th AE layer ........................................................
N_ae2=20;                                               % number of neurons of autoencoder layer
lambda_ae2=0.001;                                       % regularization parameter of autoencoder
Wb_ae2=2*rand(size(Xtr_new1,2)+1,N_ae2)-1;
X_b2 = [Xtr_new1 bias_coeff * ones(N_train,1)];
H2 =logsig(X_b2 * Wb_ae2); % tanh, logsig,mapmin-max .. % hidden layer output generation g(xw+b)
if ELMAE_selection==1 % (1)
    A = lambda_ae2*eye(size(H2,2),size(H2,2))+(H2'*H2); % (lambda * I + H'H)^-1 *H'*X
    D =  H2'*Xtr_new1;
    Beta_ae2 =pinv(A)*D;                                % Moore-Penrose pseudoinverse
else       % (2)
    q=1; iter=100;     % SIFTA parameters
    Beta_ae2  =  sparse_elm_autoencoder(H2,Xtr_new1,q,iter);
end
Xtr_new2=logsig(Xtr_new1 * Beta_ae2');
[Xtr_new2 , PS_k2]= mapminmax(Xtr_new2' , 0 ,1);        % normalization on features
Xtr_new2=Xtr_new2';
%% ELM Classifier Training  -----------------------------------------------
N_classifier=200;                                       % number of neurons of ELM classifier/regressor
lambda_cls=0.000001;                                    % regularization parameter of classifier
Wb_cls=2*rand(N_ae2+1,N_classifier)-1;
X_cla=[Xtr_new2 bias_coeff * ones(size(Xtr_new2,1),1)];
H=logsig(X_cla * Wb_cls);                               % hidden layer output generation g(xw+b)
Bata_cls=( H'*H+ (lambda_cls*eye(size(H',1))) ) \ ( H'*train_targ);
% Calculate the training accuracy -----------------------------------------
Tr_out = H * Bata_cls;
[~ ,train_estimated]= max( Tr_out ,[], 2 );
Confmat=confusionmat(Train_B0.target(:,3),train_estimated);
Training_Accuracy=sum(diag(Confmat))/sum(sum(Confmat))*100;
%% Test    ----------------------------------------------------------------
% (1) Feature mapping using the trained autoencoder 
Xtst_new1 = logsig(Test_B0.Data * Beta_ae1');           % new test data from 1-th AE
Xtst_new1= mapminmax('apply',Xtst_new1' , PS_k1);
Xtst_new1=Xtst_new1';

Xtst_new2 = logsig(Xtst_new1 * Beta_ae2');              % new test data from 1-th AE
Xtst_new2= mapminmax('apply',Xtst_new2' , PS_k2);
Xtst_new2=Xtst_new2';

% (2) Classification 
X_cls=[Xtst_new2 bias_coeff * ones(size(Xtst_new2,1),1)];
H_tst= logsig(X_cls * Wb_cls);                          % passing from activation function
Tst_out=H_tst * Bata_cls;
[~ ,test_estimated]= max( Tst_out ,[], 2 );
Confmat_tst=confusionmat(Test_B0.target(:,3),test_estimated);
Test_Accuracy=sum(diag(Confmat_tst))/sum(sum(Confmat_tst))*100;
%% Results 

Varname1l={'train ','test'};
Accuracy=[Training_Accuracy,Test_Accuracy]';
T1=table(Accuracy,'RowNames',Varname1l);disp(T1);
% figure()
confusionchart(Train_B0.target(:,3),train_estimated);title('Train');
% figure()
confusionchart(Test_B0.target(:,3),test_estimated);title('Test');