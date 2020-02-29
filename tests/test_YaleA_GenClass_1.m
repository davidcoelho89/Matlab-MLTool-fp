%% Machine Learning ToolBox

% Test For Yale A Images DataBase and others classifiers
% Author: David Nascimento Coelho
% Last Update: 2019/08/12

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% GENERAL DEFINITIONS

% General options' structure

OPT.prob = 03;              % Which problem will be solved / used
OPT.prob2 = 30;             % More details about a specific data set
OPT.norm = 03;              % Normalization definition
OPT.lbl = 01;               % Labeling definition
OPT.Nr = 50;                % Number of repetitions of the algorithm
OPT.hold = 02;              % Hold out method
OPT.ptrn = 0.70;            % Percentage of samples for training
OPT.file = 'k2nn_tst3.mat'; % file where all the variables will be saved

% Cross Validation hiperparameters

CVp.on = 1;                 % If 1, includes cross validation
CVp.fold = 5;               % Number of folds for cross validation
CVp.beta = 0.5;             % weight of number of prototypes

%% DATA LOADING AND PRE-PROCESSING

DATA = data_class_loading(OPT);     % Load Data Set
DATA = normalize(DATA,OPT);         % normalize the attributes' matrix
DATA = label_encode(DATA,OPT);      % adjust labels for the problem

%% HYPERPARAMETERS - DEFAULT

% GAUSSIAN

% ToDo - All

% OLS

OLSp_acc = cell(OPT.Nr,1);      % Init of Acc Hyperparameters of OLS
OLSp.on = 1;                    % Run the classifier
OLSp.aprox = 1;                 % Type of aproximation

% % MLP
% 
% MLPp_acc = cell(OPT.Nr,1);      % Init of Acc Hyperparameters of OLS
% MLPp.on = 1;                    % Run the classifier
% MLPp.Nh = 10;                   % No. de neuronios na camada oculta
% MLPp.Ne = 200;                  % No máximo de epocas de treino
% MLPp.eta = 0.05;                % Passo de aprendizagem
% MLPp.mom = 0.75;                % Fator de momento
% MLPp.Nlin = 2;                  % Nao-linearidade MLP (tg hiperb)

% % SVC
% 
% SVCp_acc = cell(OPT.Nr,1);      % Init of Acc Hyperparameters of OLS
% SVCp.on = 1;                    % Run the classifier
% SVCp.C = 5;                     % constante de regularização
% SVCp.Ktype = 1;                 % kernel gaussiano (tipo = 1)
% SVCp.sig2 = 0.01;               % Variancia (kernel gaussiano)
% 
% % LSSVC
% 
% LSSVCp_acc = cell(OPT.Nr,1);	% Init of Acc Hyperparameters of OLS
% LSSVCp.on = 1;                  % Run the classifier
% LSSVCp.C = 0.5;                 % constante de regularização
% LSSVCp.Ktype = 1;               % kernel gaussiano (tipo = 1)
% LSSVCp.sig2 = 128;              % Variancia (kernel gaussiano)

%% HIPERPARAMETERS - GRID FOR CROSS VALIDATION

% Get Default Hyperparameters

OLScv = OLSp;

% MLPcv = MLPp;

% SVCcv = SVCp;

% LSSVCcv = LSSVCp;

% Set Variable HyperParameters

if CVp.on == 1

OLScv;

% MLPcv.Nh = 2:20;

% SVCcv.C = [0.5 5 10 15 25 50 100 250 500 1000];
% SVCcv.sig2 = [0.01 0.05 0.1 0.5 1 5 10 50 100 500];
%     
% LSSVCcv.C = 2.^linspace(-5,20,26);
% LSSVCcv.sig2 = 2.^linspace(-10,10,21);

end

%% ACCUMULATORS

% Acc labels and data division

hold_acc = cell(OPT.Nr,1);

% Acc statistics from train and test samples

ols_tr_acc = cell(OPT.Nr,1);   	% Acc of Statistics of training data
ols_ts_acc = cell(OPT.Nr,1);   	% Acc of Statistics of test data

% mlp_tr_acc = cell(OPT.Nr,1);   	% Acc of Statistics of training data
% mlp_ts_acc = cell(OPT.Nr,1);   	% Acc of Statistics of test data
% 
% svc_tr_acc = cell(OPT.Nr,1);   	% Acc of Statistics of training data
% svc_ts_acc = cell(OPT.Nr,1);   	% Acc of Statistics of test data
% 
% lssvc_tr_acc = cell(OPT.Nr,1);	% Acc of Statistics of training data
% lssvc_ts_acc = cell(OPT.Nr,1);	% Acc of Statistics of test data

%% HOLD OUT / TRAINING / TEST / STATISTICS

disp('Begin Algorithm');

for r = 1:OPT.Nr

% %%%%%%%%% DISPLAY REPETITION AND DURATION %%%%%%%%%%%%%%

display(r);
display(datestr(now));

% %%%%%%%%%%%%%%%%%%%% HOLD OUT %%%%%%%%%%%%%%%%%%%%%%%%%%

[DATAho] = hold_out(DATA,OPT);  % Hold Out Function

DATAtr = DATAho.DATAtr;         % Training Data
DATAts = DATAho.DATAts;         % Test Data

hold_acc{r} = DATAho;           % Data Accumulator

% %%%%%%%%%%%%%%%% CROSS VALIDATION %%%%%%%%%%%%%%%%%%%%%%

% [OLSp] = cross_valid_gs(DATAtr,CVp,OLScv,@ols_train,@ols_classify);
% 
% [MLPp] = cross_valid_gs(DATAtr,CVp,MLPcv,@mlp_train,@mlp_classify);
% 
% [SVCp] = cross_valid_gs(DATAtr,CVp,SVCcv,@svc_train,@svc_classify);
% 
% [LSSVCp] = cross_valid_gs(DATAtr,CVp,LSSVCcv,@lssvc_train,@lssvc_classify);

% %%%%%%%%%%%%%% CLASSIFIERS' TRAINING %%%%%%%%%%%%%%%%%%%

[OLSp] = ols_train(DATAtr,OLSp);        % Calculate parameters
OLSp_acc{r} = OLSp;                   	% Acc Parameters

% [MLPp] = mlp_train(DATAtr,MLPp);     	% Calculate parameters
% MLPp_acc{r} = MLPp;                  	% Acc Parameters

% [SVCp] = svc_train(DATAtr,SVCp);     	% Calculate parameters
% SVCp_acc{r} = SVCp;                  	% Acc Parameters
% 
% [LSSVCp] = lssvc_train(DATAtr,LSSVCp);	% Calculate parameters
% LSSVCp_acc{r} = LSSVCp;                	% Acc Parameters

% %%%%%%%%% CLASSIFIER'S TEST AND STATISTICS %%%%%%%%%%%%%

% OLS

[OUTtr] = ols_classify(DATAtr,OLSp);            % Outputs with train data
[STATS_tr] = class_stats_1turn(DATAtr,OUTtr);	% Results with train data
ols_tr_acc{r} = STATS_tr;                       % Acc Training Statistics

[OUTts] = ols_classify(DATAts,OLSp);            % Outputs with test data
[STATS_ts] = class_stats_1turn(DATAts,OUTts);   % Results with test data
ols_ts_acc{r} = STATS_ts;                       % Acc Test Statistics

% % MLP
% 
% [OUTtr] = mlp_classify(DATAtr,MLPp);            % Outputs with train data
% [STATS_tr] = class_stats_1turn(DATAtr,OUTtr);	% Results with train data
% mlp_tr_acc{r} = STATS_tr;                       % Acc Training Statistics
% 
% [OUTts] = mlp_classify(DATAts,MLPp);            % Outputs with test data
% [STATS_ts] = class_stats_1turn(DATAts,OUTts);   % Results with test data
% mlp_ts_acc{r} = STATS_ts;                       % Acc Test Statistics

% % SVC
% 
% [OUTtr] = svc_classify(DATAtr,SVCp);            % Outputs with train data
% [STATS_tr] = class_stats_1turn(DATAtr,OUTtr);	% Results with train data
% svc_tr_acc{r} = STATS_tr;                       % Acc Training Statistics
% 
% [OUTts] = svc_classify(DATAts,SVCp);            % Outputs with test data
% [STATS_ts] = class_stats_1turn(DATAts,OUTts);   % Results with test data
% svc_ts_acc{r} = STATS_ts;                       % Acc Test Statistics
% 
% % LSSVC
% 
% [OUTtr] = lssvc_classify(DATAtr,LSSVCp);     	% Outputs with train data
% [STATS_tr] = class_stats_1turn(DATAtr,OUTtr);	% Results with train data
% lssvc_tr_acc{r} = STATS_tr;                   	% Acc Training Statistics
% 
% [OUTts] = lssvc_classify(DATAts,LSSVCp);      	% Outputs with test data
% [STATS_ts] = class_stats_1turn(DATAts,OUTts);   % Results with test data
% lssvc_ts_acc{r} = STATS_ts;                     % Acc Test Statistics

end

%% RESULTS / STATISTICS / GRAPHICS

% Statistics for n turns

nSTATS_tr_ols = class_stats_nturns(ols_tr_acc);
nSTATS_ts_ols = class_stats_nturns(ols_ts_acc);

% nSTATS_tr_mlp = class_stats_nturns(mlp_tr_acc);
% nSTATS_ts_mlp = class_stats_nturns(mlp_ts_acc);

% nSTATS_tr_svm = class_stats_nturns(svc_tr_acc);
% nSTATS_ts_svm = class_stats_nturns(svc_ts_acc);
% 
% nSTATS_tr_lssvm = class_stats_nturns(lssvc_tr_acc);
% nSTATS_ts_lssvm = class_stats_nturns(lssvc_ts_acc);

% Boxplots Init

labels = {};

Mat_boxplot1 = []; % Train Multiclass
Mat_boxplot2 = []; % Test Multiclass

% OLS

[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'OLS'};
Mat_boxplot1 = [Mat_boxplot1 nSTATS_tr_ols.acc'];
Mat_boxplot2 = [Mat_boxplot2 nSTATS_ts_ols.acc'];

% MLP

% [~,n_labels] = size(labels);
% n_labels = n_labels+1;
% labels(1,n_labels) = {'MLP'};
% Mat_boxplot1 = [Mat_boxplot1 nSTATS_tr_mlp.acc'];
% Mat_boxplot2 = [Mat_boxplot2 nSTATS_ts_mlp.acc'];

% % SVC
% 
% [~,n_labels] = size(labels);
% n_labels = n_labels+1;
% labels(1,n_labels) = {'SVC'};
% Mat_boxplot1 = [Mat_boxplot1 nSTATS_tr_svm.acc'];
% Mat_boxplot2 = [Mat_boxplot2 nSTATS_ts_svm.acc'];
% 
% % LSSVC
% 
% [~,n_labels] = size(labels);
% n_labels = n_labels+1;
% labels(1,n_labels) = {'LSSVC'};
% Mat_boxplot1 = [Mat_boxplot1 nSTATS_tr_lssvm.acc'];
% Mat_boxplot2 = [Mat_boxplot2 nSTATS_ts_lssvm.acc'];

%% GRAPHICS - DISPLAY

% BOXPLOT 1

figure; boxplot(Mat_boxplot1, 'label', labels);
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
ylabel('Accuracy')              % label eixo y
xlabel('Classifiers')           % label eixo x
title('Classifiers Comparison') % Titulo
axis ([0 n_labels+1 0 1.05])	% Eixos

hold on
media1 = mean(Mat_boxplot1);    % Taxa de acerto média
plot(media1,'*k')
hold off

% BOXPLOT 2

figure; boxplot(Mat_boxplot2, 'label', labels);
set(gcf,'color',[1 1 1])        % Tira o fundo Cinza do Matlab
ylabel('Accuracy')              % label eixo y
xlabel('Classifiers')           % label eixo x
title('Classifiers Comparison') % Titulo
axis ([0 n_labels+1 0 1.05])	% Eixos

hold on
media2 = mean(Mat_boxplot2);    % Taxa de acerto média
plot(media2,'*k')
hold off

%% SAVE DATA

% save(OPT.file);

%% END