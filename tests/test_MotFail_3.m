function [] = test_MotFail_3(OPT,CVp,REJp)

% --- Function Used to Run Different Types of Motor Combinantions ---

CVp.test = 1;   % ToDo - remove this line and use cross validation

%% DATA LOADING AND PRE-PROCESSING

DATA = data_motor_gen(OPT);         % generate data for motor problem
DATA = normalize(DATA,OPT);         % normalize the attributes' matrix
DATA = label_encode(DATA,OPT);      % adjust labels for the problem

%% HIPERPARAMETERS - DEFAULT

OLSp.on = 1;                % Run the classifier
OLSp.aprox = 1;             % Type of aproximation

BAYp.on = 1;                % Run the classifier
BAYp.type = 2;              % Type of classificer

PSp.on = 1;                 % Run the classifier
PSp.Ne = 300;               % maximum number of training epochs
PSp.eta = 0.05;             % Learning step

MLPp.on = 1;                % Run the classifier
MLPp.Nh = 10;               % No. de neuronios na camada oculta
MLPp.Ne = 200;              % No máximo de epocas de treino
MLPp.eta = 0.05;            % Passo de aprendizagem
MLPp.mom = 0.75;            % Fator de momento
MLPp.Nlin = 2;              % Nao-linearidade MLP (tg hiperb)

ELMp.on = 1;                % Run the classifier
ELMp.Nh = 25;               % No. de neuronios na camada oculta
ELMp.Nlin = 2;              % Não linearidade ELM (tg hiperb)

SVMp.on = 1;                % Run the classifier
SVMp.C = 5;                 % constante de regularização
SVMp.Ktype = 1;             % kernel gaussiano (tipo = 1)
SVMp.sig2 = 0.01;           % Variancia (kernel gaussiano)

LSSVMp.on = 1;              % Run the classifier
LSSVMp.C = 0.5;             % constante de regularização
LSSVMp.Ktype = 1;           % kernel gaussiano (tipo = 1)
LSSVMp.sig2 = 128;          % Variancia (kernel gaussiano)

MLMp.on = 1;                % Run the classifier
MLMp.K = 09;                % Number of reference points

GPp.on = 1;                 % Run the classifier
GPp.l2 = 2;                 % Constante GP1
GPp.K = 1;                  % kernel gaussiano (tipo = 1)
GPp.sig2 = 2;               % Constante GP2

%% HIPERPARAMETERS - GRID SEARCH FOR CROSS VALIDATION

% OLScv = OLSp;                           % Constant Hyperparameters
% % ToDo - type of aproximation
% 
% BAYcv = BAYp;                           % Constant Hyperparameters
% % ToDo - type of classifier
% 
% PScv = PSp;                             % Constant Hyperparameters
% % ToDo - learning step
% 
% MLPcv = MLPp;                           % Constant Hyperparameters
% MLPcv.Nh = 2:20;                        % Number of hidden neurons
% mlp_Nh = zeros(OPT.Nr,1);               % Acc number of hidden neurons
% 
% ELMcv = ELMp;                           % Constant Hyperparameters
% ELMcv.Nh = 10:30;                       % Number of hidden neurons
% elm_Nh = zeros(OPT.Nr,1);               % Acc number of hidden neurons
% 
% SVMcv = SVMp;                           % Constant Hyperparameters
% SVMcv.C = [0.5 5 10 15 25 50 100 250 500 1000];
% SVMcv.sig2 = [0.01 0.05 0.1 0.5 1 5 10 50 100 500];
% svm_C = zeros(OPT.Nr,1);                % Acc regularization constant
% svm_sig2 = zeros(OPT.Nr,1);             % Acc of variance(gaussian kernel)
% svm_nsv = zeros(OPT.Nr,1);              % Acc number of support vectors
% 
% LSSVMcv = LSSVMp;                       % Constant Hyperparameters
% LSSVMcv.C = 2.^linspace(-5,20,26);
% LSSVMcv.sig2 = 2.^linspace(-10,10,21);
% lssvm_C = zeros(OPT.Nr,1);              % Acc regularization constant
% lssvm_sig2 = zeros(OPT.Nr,1);           % Acc of variance(gaussian kernel)
% lssvm_nsv = zeros(OPT.Nr,1);            % Acc number of support vectors
% 
% MLMcv = MLMp;                           % Constant Hyperparameters
% MLMcv.K = 2:15;                         % Reference points
% mlm_K = zeros(OPT.Nr,1);                % Acc number of reference points

%% CLASSIFIERS' RESULTS INIT

DATA_acc = cell(OPT.Nr,1);    	% Acc data division

ols_out_tr = cell(OPT.Nr,1);	% Acc of training data output
ols_out_ts = cell(OPT.Nr,1);	% Acc of test data output
ols_out_rj = cell(OPT.Nr,1);	% Acc of reject option output
ols_Mconf_sum = zeros(Nc,Nc);   % Aux var for mean confusion matrix calc

bay_out_tr = cell(OPT.Nr,1);	% Acc of training data output
bay_out_ts = cell(OPT.Nr,1);	% Acc of test data output
bay_out_rj = cell(OPT.Nr,1);	% Acc of reject option output
bay_Mconf_sum = zeros(Nc,Nc);   % Aux var for mean confusion matrix calc

ps_out_tr = cell(OPT.Nr,1);     % Acc of training data output
ps_out_ts = cell(OPT.Nr,1);     % Acc of test data output
ps_out_rj = cell(OPT.Nr,1);     % Acc of reject option output
ps_Mconf_sum = zeros(Nc,Nc);    % Aux var for mean confusion matrix calc

mlp_out_tr = cell(OPT.Nr,1);	% Acc of training data output
mlp_out_ts = cell(OPT.Nr,1);	% Acc of test data output
mlp_out_rj = cell(OPT.Nr,1);	% Acc of reject option output
mlp_Mconf_sum = zeros(Nc,Nc);   % Aux var for mean confusion matrix calc

elm_out_tr = cell(OPT.Nr,1);	% Acc of training data output
elm_out_ts = cell(OPT.Nr,1);	% Acc of test data output
elm_out_rj = cell(OPT.Nr,1);	% Acc of reject option output
elm_Mconf_sum = zeros(Nc,Nc);   % Aux var for mean confusion matrix calc

svm_out_tr = cell(OPT.Nr,1);	% Acc of training data output
svm_out_ts = cell(OPT.Nr,1);	% Acc of test data output
svm_out_rj = cell(OPT.Nr,1);	% Acc of reject option output
svm_Mconf_sum = zeros(Nc,Nc);   % Aux var for mean confusion matrix calc

lssvm_out_tr = cell(OPT.Nr,1);	% Acc of training data output
lssvm_out_ts = cell(OPT.Nr,1);	% Acc of test data output
lssvm_out_rj = cell(OPT.Nr,1);	% Acc of reject option output
lssvm_Mconf_sum = zeros(Nc,Nc); % Aux var for mean confusion matrix calc

mlm_out_tr = cell(OPT.Nr,1);	% Acc of training data output
mlm_out_ts = cell(OPT.Nr,1);	% Acc of test data output
mlm_out_rj = cell(OPT.Nr,1);	% Acc of reject option output
mlm_Mconf_sum = zeros(Nc,Nc);   % Aux var for mean confusion matrix calc

%% HOLD OUT / CROSS VALIDATION / TRAINING / TEST

for r = 1:OPT.Nr,

% Display, at Command Window, each repeat

display(r);
display(datestr(now));

% %%%%%%%%%%%%%%%%%%%% HOLD OUT %%%%%%%%%%%%%%%%%%%%%%%%%%
    
DATA_acc{r} = hold_out(DATA,OPT);   % Save data division
DATAtr = DATA_acc{r}.DATAtr;        % Training Data
DATAts = DATA_acc{r}.DATAts;      	% Test Data

% %%%%%%%%%%%%%% SHUFFLE TRAINING DATA %%%%%%%%%%%%%%%%%%%

I = randperm(size(DATAtr.input,2));
DATAtr.input = DATAtr.input(:,I);
DATAtr.output = DATAtr.output(:,I);
DATAtr.lbl = DATAtr.lbl(:,I);

% %%%%%%%%%%%%%%%% CROSS VALIDATION %%%%%%%%%%%%%%%%%%%%%%

% OLS - ToDo - All

% BAYES - ToDo - All

% PS - ToDo - All

% [MLPp] = mlp_cv(DATAtr,MLPp,MLPcv,CVp);
% mlp_Nh(r) = MLPp.Nh;
% 
% [ELMp] = elm_cv(DATAtr,ELMp,ELMcv,CVp);
% elm_Nh(r) = ELMp.Nh;
% 
% [SVMp] = svm_cv(DATAtr,SVMp,SVMcv,CVp);
% svm_C(r) = SVMp.C;
% svm_sig2(r) = SVMp.sig2;
% 
% [LSSVMp] = lssvm_cv(DATAtr,LSSVMp,LSSVMcv,CVp);
% lssvm_C(r) = LSSVMp.C;
% lssvm_sig2(r) = LSSVMp.sig2;
% 
% [MLMp] = mlm_cv(DATAtr,MLMp,MLMcv,CVp);
% mlm_K(r) = MLMp.K;

% GP - ToDo - All

% %%%%%%%%%%%%%% CLASSIFIERS' TRAINING %%%%%%%%%%%%%%%%%%%

[OLSp] = ols_train(DATAtr,OLSp);

[BAYp] = gauss_train(DATAtr,BAYp);

[PSp] = ps_train(DATAtr,PSp);

[MLPp] = mlp_train(DATAtr,MLPp);
 
[ELMp] = elm_train(DATAtr,ELMp);

[SVMp] = svm_train(DATAtr,SVMp);
 
[LSSVMp] = lssvm_train(DATAtr,LSSVMp);
 
[MLMp] = mlm_train(DATAtr,MLMp);
 
[GPp] = gp_train(DATAtr,GPp);

% %%%%%%%%%%%%%%%%% CLASSIFIERS' TEST %%%%%%%%%%%%%%%%%%%%

% OLS

[OUTtr] = ols_classify(DATAtr,OLSp);
OUTtr.nf = normal_or_fail(OUTtr.Mconf);
OUTtr.mcc = Mcc(OUTtr.Mconf);
ols_out_tr{r} = OUTtr;

[OUTts] = ols_classify(DATAts,OLSp);
OUTts.nf = normal_or_fail(OUTts.Mconf);
OUTts.mcc = Mcc(OUTts.Mconf);
ols_out_ts{r} = OUTts;

[OUTrj] = reject_opt2(DATAts,OUTts,REJp);
OUTrj.nf = normal_or_fail(OUTrj.Mconf);
OUTrj.mcc = Mcc(OUTrj.Mconf);
ols_out_rj{r} = OUTrj;

ols_Mconf_sum = ols_Mconf_sum + OUTts.Mconf;

% BAYES

[OUTtr] = gauss_classify(DATAtr,BAYp);
OUTtr.nf = normal_or_fail(OUTtr.Mconf);
OUTtr.mcc = Mcc(OUTtr.Mconf);
bay_out_tr{r,1} = OUTtr;

[OUTts] = gauss_classify(DATAts,BAYp);
OUTts.nf = normal_or_fail(OUTts.Mconf);
OUTts.mcc = Mcc(OUTts.Mconf);
bay_out_ts{r,1} = OUTts;

[OUTrj] = reject_opt2(DATAts,OUTts,REJp);
OUTrj.nf = normal_or_fail(OUTrj.Mconf);
OUTrj.mcc = Mcc(OUTrj.Mconf);
bay_out_rj{r,1} = OUTrj;

bay_Mconf_sum = bay_Mconf_sum + OUTts.Mconf;

% PS

[OUTtr] = ps_classify(DATAtr,PSp);
OUTtr.nf = normal_or_fail(OUTtr.Mconf);
OUTtr.mcc = Mcc(OUTtr.Mconf);
ps_out_tr{r,1} = OUTtr;

[OUTts] = ps_classify(DATAts,PSp);
OUTts.nf = normal_or_fail(OUTts.Mconf);
OUTts.mcc = Mcc(OUTts.Mconf);
ps_out_ts{r,1} = OUTts;

[OUTrj] = reject_opt2(DATAts,OUTts,REJp);
OUTrj.nf = normal_or_fail(OUTrj.Mconf);
OUTrj.mcc = Mcc(OUTrj.Mconf);
ps_out_rj{r,1} = OUTrj;

ps_Mconf_sum = ps_Mconf_sum + OUTts.Mconf;

% MLP

[OUTtr] = mlp_classify(DATAtr,MLPp);
OUTtr.nf = normal_or_fail(OUTtr.Mconf);
OUTtr.mcc = Mcc(OUTtr.Mconf);
mlp_out_tr{r,1} = OUTtr;

[OUTts] = mlp_classify(DATAts,MLPp);
OUTts.nf = normal_or_fail(OUTts.Mconf);
OUTts.mcc = Mcc(OUTts.Mconf);
mlp_out_ts{r,1} = OUTts;

[OUTrj] = reject_opt2(DATAts,OUTts,REJp);
OUTrj.nf = normal_or_fail(OUTrj.Mconf);
OUTrj.mcc = Mcc(OUTrj.Mconf);
mlp_out_rj{r,1} = OUTrj;

mlp_Mconf_sum = mlp_Mconf_sum + OUTts.Mconf;

% ELM

[OUTtr] = elm_classify(DATAtr,ELMp);
OUTtr.nf = normal_or_fail(OUTtr.Mconf);
OUTtr.mcc = Mcc(OUTtr.Mconf);
elm_out_tr{r,1} = OUTtr;

[OUTts] = elm_classify(DATAts,ELMp);
OUTts.nf = normal_or_fail(OUTts.Mconf);
OUTts.mcc = Mcc(OUTts.Mconf);
elm_out_ts{r,1} = OUTts;

[OUTrj] = reject_opt2(DATAts,OUTts,REJp);
OUTrj.nf = normal_or_fail(OUTrj.Mconf);
OUTrj.mcc = Mcc(OUTrj.Mconf);
elm_out_rj{r,1} = OUTrj;

elm_Mconf_sum = elm_Mconf_sum + OUTts.Mconf;

% SVM

[OUTtr] = svm_classify(DATAtr,SVMp);
OUTtr.nf = normal_or_fail(OUTtr.Mconf);
OUTtr.mcc = Mcc(OUTtr.Mconf);
svm_out_tr{r,1} = OUTtr;

[OUTts] = svm_classify(DATAts,SVMp);
OUTts.nf = normal_or_fail(OUTts.Mconf);
OUTts.mcc = Mcc(OUTts.Mconf);
svm_out_ts{r,1} = OUTts;

[OUTrj] = reject_opt2(DATAts,OUTts,REJp);
OUTrj.nf = normal_or_fail(OUTrj.Mconf);
OUTrj.mcc = Mcc(OUTrj.Mconf);
svm_out_rj{r,1} = OUTrj;

svm_Mconf_sum = svm_Mconf_sum + OUTts.Mconf;

% LSSVM

[OUTtr] = lssvm_classify(DATAtr,LSSVMp);
OUTtr.nf = normal_or_fail(OUTtr.Mconf);
OUTtr.mcc = Mcc(OUTtr.Mconf);
lssvm_out_tr{r,1} = OUTtr;

[OUTts] = lssvm_classify(DATAts,LSSVMp);
OUTts.nf = normal_or_fail(OUTts.Mconf);
OUTts.mcc = Mcc(OUTts.Mconf);
lssvm_out_ts{r,1} = OUTts;

[OUTrj] = reject_opt2(DATAts,OUTts,REJp);
OUTrj.nf = normal_or_fail(OUTrj.Mconf);
OUTrj.mcc = Mcc(OUTrj.Mconf);
lssvm_out_rj{r,1} = OUTrj;

lssvm_Mconf_sum = lssvm_Mconf_sum + OUTts.Mconf;

% MLM

[OUTtr] = mlm_classify(DATAtr,MLMp);
OUTtr.nf = normal_or_fail(OUTtr.Mconf);
mlm_out_tr{r,1} = OUTtr;

[OUTts] = mlm_classify(DATAts,MLMp);
OUTts.nf = normal_or_fail(OUTts.Mconf);
OUTts.mcc = Mcc(OUTts.Mconf);
mlm_out_ts{r,1} = OUTts;

[OUTrj] = reject_opt2(DATAts,OUTts,REJp);
OUTrj.nf = normal_or_fail(OUTrj.Mconf);
OUTrj.mcc = Mcc(OUTrj.Mconf);
mlm_out_rj{r,1} = OUTrj;

mlm_Mconf_sum = mlm_Mconf_sum + OUTts.Mconf;

end

%% ESTATISTICAS



%% GERAÇÃO DOS GRAFICOS

% Inicializa célula de labels e matriz de acertos para testes
labels = {};
Mat_boxplot1 = [];
Mat_boxplot2 = [];
Mat_boxplot3 = [];
Mat_boxplot4 = [];
Mat_boxplot5 = [];
Mat_boxplot6 = [];

if OLSp.on == 1,
[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'OLS'};
Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(ols_out_tr)];
Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(ols_out_tr)];
Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(ols_out_ts)];
Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(ols_out_ts)];
Mat_boxplot5 = [Mat_boxplot5 accuracy_mult(ols_out_rj)];
Mat_boxplot6 = [Mat_boxplot6 accuracy_bin(ols_out_rj)];
end

if BAYp.on == 1,
[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'BAY'};
Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(bay_out_tr)];
Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(bay_out_tr)];
Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(bay_out_ts)];
Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(bay_out_ts)];
Mat_boxplot5 = [Mat_boxplot5 accuracy_mult(bay_out_rj)];
Mat_boxplot6 = [Mat_boxplot6 accuracy_bin(bay_out_rj)];
end

if PSp.on == 1,
[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'PS'};
Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(ps_out_tr)];
Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(ps_out_tr)];
Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(ps_out_ts)];
Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(ps_out_ts)];
Mat_boxplot5 = [Mat_boxplot5 accuracy_mult(ps_out_rj)];
Mat_boxplot6 = [Mat_boxplot6 accuracy_bin(ps_out_rj)];
end

if MLPp.on == 1,
[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'MLP'};
Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(mlp_out_tr)];
Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(mlp_out_tr)];
Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(mlp_out_ts)];
Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(mlp_out_ts)];
Mat_boxplot5 = [Mat_boxplot5 accuracy_mult(mlp_out_rj)];
Mat_boxplot6 = [Mat_boxplot6 accuracy_bin(mlp_out_rj)];
end

if ELMp.on == 1,
[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'ELM'};
Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(elm_out_tr)];
Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(elm_out_tr)];
Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(elm_out_ts)];
Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(elm_out_ts)];
Mat_boxplot5 = [Mat_boxplot5 accuracy_mult(elm_out_rj)];
Mat_boxplot6 = [Mat_boxplot6 accuracy_bin(elm_out_rj)];
end

if SVMp.on == 1,
[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'SVM'};
Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(svm_out_tr)];
Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(svm_out_tr)];
Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(svm_out_ts)];
Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(svm_out_ts)];
Mat_boxplot5 = [Mat_boxplot5 accuracy_mult(svm_out_rj)];
Mat_boxplot6 = [Mat_boxplot6 accuracy_bin(svm_out_rj)];
end

if LSSVMp.on == 1,
[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'LSSVM'};
Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(lssvm_out_tr)];
Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(lssvm_out_tr)];
Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(lssvm_out_ts)];
Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(lssvm_out_ts)];
Mat_boxplot5 = [Mat_boxplot5 accuracy_mult(lssvm_out_rj)];
Mat_boxplot6 = [Mat_boxplot6 accuracy_bin(lssvm_out_rj)];
end

if MLMp.on == 1,
[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'MLM'};
Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(mlm_out_tr)];
Mat_boxplot2 = [Mat_boxplot2 accuracy_bin(mlm_out_tr)];
Mat_boxplot3 = [Mat_boxplot3 accuracy_mult(mlm_out_ts)];
Mat_boxplot4 = [Mat_boxplot4 accuracy_bin(mlm_out_ts)];
Mat_boxplot5 = [Mat_boxplot5 accuracy_mult(mlm_out_rj)];
Mat_boxplot6 = [Mat_boxplot6 accuracy_bin(mlm_out_rj)];
end

% BOXPLOT 1
figure; boxplot(Mat_boxplot1, 'label', labels);
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
ylabel('Acurácia')              % label eixo y
xlabel('Classificadores')       % label eixo x
title('Taxa de Classificação')  % Titulo
axis ([0 n_labels+1 0 1.05])	% Eixos

hold on
media1 = mean(Mat_boxplot1);    % Taxa de acerto média
plot(media1,'*k')
hold off

% BOXPLOT 2
figure; boxplot(Mat_boxplot2, 'label', labels);
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
ylabel('Acurácia')              % label eixo y
xlabel('Classificadores')       % label eixo x
title('Taxa de Classificação')  % Titulo
axis ([0 n_labels+1 0 1.05])	% Eixos

hold on
media2 = mean(Mat_boxplot2);    % Taxa de acerto média
plot(media2,'*k')
hold off

% BOXPLOT 3
figure; boxplot(Mat_boxplot3, 'label', labels);
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
ylabel('Acurácia')              % label eixo y
xlabel('Classificadores')       % label eixo x
title('Taxa de Classificação')  % Titulo
axis ([0 n_labels+1 0 1.05])	% Eixos

hold on
media3 = mean(Mat_boxplot3);    % Taxa de acerto média
plot(media3,'*k')
hold off

% BOXPLOT 4
figure; boxplot(Mat_boxplot4, 'label', labels);
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
ylabel('Acurácia')              % label eixo y
xlabel('Classificadores')       % label eixo x
title('Taxa de Classificação')  % Titulo
axis ([0 n_labels+1 0 1.05])	% Eixos

hold on
media4 = mean(Mat_boxplot4);    % Taxa de acerto média
plot(media4,'*k')
hold off

% BOXPLOT 5
figure; boxplot(Mat_boxplot5, 'label', labels);
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
ylabel('Acurácia')              % label eixo y
xlabel('Classificadores')       % label eixo x
title('Taxa de Classificação')  % Titulo
axis ([0 n_labels+1 0 1.05])	% Eixos

hold on
media3 = mean(Mat_boxplot5);    % Taxa de acerto média
plot(media3,'*k')
hold off

% BOXPLOT 6
figure; boxplot(Mat_boxplot6, 'label', labels);
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
ylabel('Acurácia')              % label eixo y
xlabel('Classificadores')       % label eixo x
title('Taxa de Classificação')  % Titulo
axis ([0 n_labels+1 0 1.05])	% Eixos

hold on
media4 = mean(Mat_boxplot6);    % Taxa de acerto média
plot(media4,'*k')
hold off

%% SALVAR DADOS

save(OPT.file);

%% END