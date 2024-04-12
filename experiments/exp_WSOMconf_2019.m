%% Machine Learning ToolBox

% ESANN 2019 Tests 3 - ALD Prototype-based Classifiers
% Author: David Nascimento Coelho
% Last Update: 2019/01/22

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% GENERAL DEFINITIONS

% General options' structure

OPT.prob = 10;                % Which problem will be solved / used
OPT.prob2 = 02;               % When it needs an specification of data set
OPT.norm = 3;                 % Normalization definition
OPT.lbl = 0;                  % Data labeling definition
OPT.Nr = 50;                  % Number of repetitions of each algorithm
OPT.hold = 01;                % Hold out method
OPT.ptrn = 0.7;               % Percentage of samples for training
OPT.file = 'res_ald_v2.mat';  % file where all the variables will be saved  

% Cross Validation hiperparameters

CVp.on = 1;                 % If 1, includes cross validation
CVp.fold = 5;               % Number of folds for cross validation
CVp.beta = 0.5;             % weight of number of prototypes

%% DATA LOADING AND PRE-PROCESSING

DATA = data_class_loading(OPT);     % Load Data Set
DATA = normalize(DATA,OPT);         % normalize the attributes' matrix
DATA = label_encode(DATA,OPT);      % adjust labels for the problem

%% HYPERPARAMETERS - DEFAULT

PAR_common.on = 1;              % Run the classifier
PAR_common.v = 0.1;           	% sparsiveness parameter

KALD1p_acc = cell(OPT.Nr,1);    % Init of Acc Hyperparameters of KALD-1
PAR_kald_1 = PAR_common;        % Get common parameters
PAR_kald_1.St = 1;              % Sparsification Type (per data set or per class)
PAR_kald_1.Ktype = 1;           % Kernel Type (lin, gauss, cauchy, log)
PAR_kald_1.sig2 = 2;            % comprehensiveness of cluster

KALD2p_acc = cell(OPT.Nr,1);    % Init of Acc Hyperparameters of KALD-2
PAR_kald_2 = PAR_common;        % Get common parameters
PAR_kald_2.St = 2;              % Sparsification Type (per data set or per class)
PAR_kald_2.Ktype = 1;           % Kernel Type (lin, gauss, cauchy, log)
PAR_kald_2.sig2 = 2;            % comprehensiveness of cluster

KALD3p_acc = cell(OPT.Nr,1);    % Init of Acc Hyperparameters of KALD-3
PAR_kald_3 = PAR_common;        % Get common parameters
PAR_kald_3.St = 1;              % Sparsification Type (per data set or per class)
PAR_kald_3.Ktype = 2;           % Kernel Type (lin, gauss, cauchy, log)
PAR_kald_3.sig2 = 2;            % comprehensiveness of cluster

KALD4p_acc = cell(OPT.Nr,1);    % Init of Acc Hyperparameters of KALD-4
PAR_kald_4 = PAR_common;        % Get common parameters
PAR_kald_4.St = 2;              % Sparsification Type (per data set or per class)
PAR_kald_4.Ktype = 2;           % Kernel Type (lin, gauss, cauchy, log)
PAR_kald_4.sig2 = 2;            % comprehensiveness of cluster

KALD5p_acc = cell(OPT.Nr,1);    % Init of Acc Hyperparameters of KALD-5
PAR_kald_5 = PAR_common;        % Get common parameters
PAR_kald_5.St = 1;              % Sparsification Type (per data set or per class)
PAR_kald_5.Ktype = 3;           % Kernel Type (lin, gauss, cauchy, log)
PAR_kald_5.sig2 = 2;            % comprehensiveness of cluster

KALD6p_acc = cell(OPT.Nr,1);    % Init of Acc Hyperparameters of KALD-6
PAR_kald_6 = PAR_common;        % Get common parameters
PAR_kald_6.St = 2;              % Sparsification Type (per data set or per class)
PAR_kald_6.Ktype = 3;           % Kernel Type (lin, gauss, cauchy, log)
PAR_kald_6.sig2 = 2;            % comprehensiveness of cluster

KALD7p_acc = cell(OPT.Nr,1);    % Init of Acc Hyperparameters of KALD-7
PAR_kald_7 = PAR_common;        % Get common parameters
PAR_kald_7.St = 1;              % Sparsification Type (per data set or per class)
PAR_kald_7.Ktype = 4;           % Kernel Type (lin, gauss, cauchy, log)
PAR_kald_7.sig2 = 2;            % comprehensiveness of cluster

KALD8p_acc = cell(OPT.Nr,1);    % Init of Acc Hyperparameters of KALD-8
PAR_kald_8 = PAR_common;        % Get common parameters
PAR_kald_8.St = 2;              % Sparsification Type (per data set or per class)
PAR_kald_8.Ktype = 4;           % Kernel Type (lin, gauss, cauchy, log)
PAR_kald_8.sig2 = 2;            % comprehensiveness of cluster

%% HIPERPARAMETERS - GRID FOR CROSS VALIDATION

% Get Default Hyperparameters

KALD1cv = PAR_kald_1;
KALD2cv = PAR_kald_2;
KALD3cv = PAR_kald_3;
KALD4cv = PAR_kald_4;
KALD5cv = PAR_kald_5;
KALD6cv = PAR_kald_6;
KALD7cv = PAR_kald_7;
KALD8cv = PAR_kald_8;

% Set Variable HyperParameters

if CVp.on == 1

KALD1cv.v = 2.^linspace(-4,3,8);

KALD2cv.v = 2.^linspace(-4,3,8);

KALD3cv.v = 2.^linspace(-4,3,8);
KALD3cv.sig2 = 2.^linspace(-10,9,20);

KALD4cv.v = 2.^linspace(-4,3,8);
KALD4cv.sig2 = 2.^linspace(-10,9,20);

KALD5cv.v = 2.^linspace(-4,3,8);
KALD5cv.sig2 = 2.^linspace(-10,9,20);

KALD6cv.v = 2.^linspace(-4,3,8);
KALD6cv.sig2 = 2.^linspace(-10,9,20);

KALD7cv.v = -2.^linspace(10,2,9);
KALD7cv.sig2 = [0.001 0.01 0.1 1 2 5];

KALD8cv.v = -2.^linspace(10,2,9);
KALD8cv.sig2 = [0.001 0.01 0.1 1 2 5];

end

%% ACCUMULATORS

hold_acc = cell(OPT.Nr,1);          % Acc of labels and data division

kald1_out_tr = cell(OPT.Nr,1);     	% Acc of training data output
kald1_out_ts = cell(OPT.Nr,1);    	% Acc of test data output
kald1_Mconf_sum = zeros(Nc,Nc);  	% Aux var for mean confusion matrix calc

kald2_out_tr = cell(OPT.Nr,1);     	% Acc of training data output
kald2_out_ts = cell(OPT.Nr,1);    	% Acc of test data output
kald2_Mconf_sum = zeros(Nc,Nc);  	% Aux var for mean confusion matrix calc

kald3_out_tr = cell(OPT.Nr,1);     	% Acc of training data output
kald3_out_ts = cell(OPT.Nr,1);    	% Acc of test data output
kald3_Mconf_sum = zeros(Nc,Nc);  	% Aux var for mean confusion matrix calc

kald4_out_tr = cell(OPT.Nr,1);     	% Acc of training data output
kald4_out_ts = cell(OPT.Nr,1);    	% Acc of test data output
kald4_Mconf_sum = zeros(Nc,Nc);  	% Aux var for mean confusion matrix calc

kald5_out_tr = cell(OPT.Nr,1);     	% Acc of training data output
kald5_out_ts = cell(OPT.Nr,1);    	% Acc of test data output
kald5_Mconf_sum = zeros(Nc,Nc);  	% Aux var for mean confusion matrix calc

kald6_out_tr = cell(OPT.Nr,1);     	% Acc of training data output
kald6_out_ts = cell(OPT.Nr,1);    	% Acc of test data output
kald6_Mconf_sum = zeros(Nc,Nc);  	% Aux var for mean confusion matrix calc

kald7_out_tr = cell(OPT.Nr,1);     	% Acc of training data output
kald7_out_ts = cell(OPT.Nr,1);    	% Acc of test data output
kald7_Mconf_sum = zeros(Nc,Nc);  	% Aux var for mean confusion matrix calc

kald8_out_tr = cell(OPT.Nr,1);     	% Acc of training data output
kald8_out_ts = cell(OPT.Nr,1);    	% Acc of test data output
kald8_Mconf_sum = zeros(Nc,Nc);  	% Aux var for mean confusion matrix calc

%% HOLD OUT / CROSS VALIDATION / TRAINING / TEST

for r = 1:OPT.Nr

% %%%%%%%%% DISPLAY REPETITION AND DURATION %%%%%%%%%%%%%%

disp(r);
display(datestr(now));

% %%%%%%%%%%%%%%%%%%%% HOLD OUT %%%%%%%%%%%%%%%%%%%%%%%%%%

[DATAho] = hold_out(DATA,OPT);

DATAtr = DATAho.DATAtr;
DATAts = DATAho.DATAts;

hold_acc{r} = DATAho;

% %%%%%%%%%%%%%%%% CROSS VALIDATION %%%%%%%%%%%%%%%%%%%%%%

[PAR_kald_1] = cross_valid_gs2(DATAtr,CVp,KALD1cv,@kald_train,@kald_classify);

[PAR_kald_2] = cross_valid_gs2(DATAtr,CVp,KALD2cv,@kald_train,@kald_classify);

[PAR_kald_3] = cross_valid_gs2(DATAtr,CVp,KALD3cv,@kald_train,@kald_classify);

[PAR_kald_4] = cross_valid_gs2(DATAtr,CVp,KALD4cv,@kald_train,@kald_classify);

[PAR_kald_5] = cross_valid_gs2(DATAtr,CVp,KALD5cv,@kald_train,@kald_classify);

[PAR_kald_6] = cross_valid_gs2(DATAtr,CVp,KALD6cv,@kald_train,@kald_classify);

[PAR_kald_7] = cross_valid_gs2(DATAtr,CVp,KALD7cv,@kald_train,@kald_classify);

[PAR_kald_8] = cross_valid_gs2(DATAtr,CVp,KALD8cv,@kald_train,@kald_classify);

% %%%%%%%%%%%%%% CLASSIFIERS' TRAINING %%%%%%%%%%%%%%%%%%%

PAR_kald_1 = kald_train(DATAtr,PAR_kald_1);
KALD1p_acc{r} = PAR_kald_1;

PAR_kald_2 = kald_train(DATAtr,PAR_kald_2);
KALD2p_acc{r} = PAR_kald_2;

PAR_kald_3 = kald_train(DATAtr,PAR_kald_3);
KALD3p_acc{r} = PAR_kald_3;

PAR_kald_4 = kald_train(DATAtr,PAR_kald_4);
KALD4p_acc{r} = PAR_kald_4;

PAR_kald_5 = kald_train(DATAtr,PAR_kald_5);
KALD5p_acc{r} = PAR_kald_5;

PAR_kald_6 = kald_train(DATAtr,PAR_kald_6);
KALD6p_acc{r} = PAR_kald_6;

PAR_kald_7 = kald_train(DATAtr,PAR_kald_7);
KALD7p_acc{r} = PAR_kald_7;

PAR_kald_8 = kald_train(DATAtr,PAR_kald_8);
KALD8p_acc{r} = PAR_kald_8;

% %%%%%%%%%%%%%%%%% CLASSIFIERS' TEST %%%%%%%%%%%%%%%%%%%%

% KALD 1 (linear)

[OUTtr] = kald_classify(DATAtr,PAR_kald_1);
kald1_out_tr{r,1} = OUTtr;

[OUTts] = kald_classify(DATAts,PAR_kald_1);
kald1_out_ts{r,1} = OUTts;

kald1_Mconf_sum = kald1_Mconf_sum + OUTts.Mconf;

% KALD 2 (linear)

[OUTtr] = kald_classify(DATAtr,PAR_kald_2);
kald2_out_tr{r,1} = OUTtr;

[OUTts] = kald_classify(DATAts,PAR_kald_2);
kald2_out_ts{r,1} = OUTts;

kald2_Mconf_sum = kald2_Mconf_sum + OUTts.Mconf;

% KALD 3 (gaussian)

[OUTtr] = kald_classify(DATAtr,PAR_kald_3);
kald3_out_tr{r,1} = OUTtr;

[OUTts] = kald_classify(DATAts,PAR_kald_3);
kald3_out_ts{r,1} = OUTts;

kald3_Mconf_sum = kald3_Mconf_sum + OUTts.Mconf;

% KALD 4 (gaussian)

[OUTtr] = kald_classify(DATAtr,PAR_kald_4);
kald4_out_tr{r,1} = OUTtr;

[OUTts] = kald_classify(DATAts,PAR_kald_4);
kald4_out_ts{r,1} = OUTts;

kald4_Mconf_sum = kald4_Mconf_sum + OUTts.Mconf;

% KALD 5 (cauchy)

[OUTtr] = kald_classify(DATAtr,PAR_kald_5);
kald5_out_tr{r,1} = OUTtr;

[OUTts] = kald_classify(DATAts,PAR_kald_5);
kald5_out_ts{r,1} = OUTts;

kald5_Mconf_sum = kald5_Mconf_sum + OUTts.Mconf;

% KALD 6 (cauchy)

[OUTtr] = kald_classify(DATAtr,PAR_kald_6);
kald6_out_tr{r,1} = OUTtr;

[OUTts] = kald_classify(DATAts,PAR_kald_6);
kald6_out_ts{r,1} = OUTts;

kald6_Mconf_sum = kald6_Mconf_sum + OUTts.Mconf;

% KALD 7 (log)

[OUTtr] = kald_classify(DATAtr,PAR_kald_7);
kald7_out_tr{r,1} = OUTtr;

[OUTts] = kald_classify(DATAts,PAR_kald_7);
kald7_out_ts{r,1} = OUTts;

kald7_Mconf_sum = kald7_Mconf_sum + OUTts.Mconf;

% KALD 8 (log)

[OUTtr] = kald_classify(DATAtr,PAR_kald_8);
kald8_out_tr{r,1} = OUTtr;

[OUTts] = kald_classify(DATAts,PAR_kald_8);
kald8_out_ts{r,1} = OUTts;

kald8_Mconf_sum = kald8_Mconf_sum + OUTts.Mconf;

end

%% STATISTICS

% Mean Confusion Matrix

kald1_Mconf_sum = kald1_Mconf_sum / OPT.Nr;
kald2_Mconf_sum = kald2_Mconf_sum / OPT.Nr;
kald3_Mconf_sum = kald3_Mconf_sum / OPT.Nr;
kald4_Mconf_sum = kald4_Mconf_sum / OPT.Nr;
kald5_Mconf_sum = kald5_Mconf_sum / OPT.Nr;
kald6_Mconf_sum = kald6_Mconf_sum / OPT.Nr;
kald7_Mconf_sum = kald7_Mconf_sum / OPT.Nr;
kald8_Mconf_sum = kald8_Mconf_sum / OPT.Nr;

%% GRAPHICS - CONSTRUCT

% Init labels' cells and Init boxplot matrix

labels = {};

Mat_boxplot1 = []; % Train Multiclass
Mat_boxplot2 = []; % Test Multiclass

% KALD-1

[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'KALD-1'};
Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(kald1_out_tr)];
Mat_boxplot2 = [Mat_boxplot2 accuracy_mult(kald1_out_ts)];

% KALD-2

[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'KALD-2'};
Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(kald2_out_tr)];
Mat_boxplot2 = [Mat_boxplot2 accuracy_mult(kald2_out_ts)];

% KALD-3

[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'KALD-G1'};
Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(kald3_out_tr)];
Mat_boxplot2 = [Mat_boxplot2 accuracy_mult(kald3_out_ts)];

% KALD-4

[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'KALD-G2'};
Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(kald4_out_tr)];
Mat_boxplot2 = [Mat_boxplot2 accuracy_mult(kald4_out_ts)];

% KALD-5

[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'KALD-C1'};
Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(kald5_out_tr)];
Mat_boxplot2 = [Mat_boxplot2 accuracy_mult(kald5_out_ts)];

% KALD-6

[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'KALD-C2'};
Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(kald6_out_tr)];
Mat_boxplot2 = [Mat_boxplot2 accuracy_mult(kald6_out_ts)];

% KALD-7

[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'KALD-L1'};
Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(kald7_out_tr)];
Mat_boxplot2 = [Mat_boxplot2 accuracy_mult(kald7_out_ts)];

% KALD-8

[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'KALD-L2'};
Mat_boxplot1 = [Mat_boxplot1 accuracy_mult(kald8_out_tr)];
Mat_boxplot2 = [Mat_boxplot2 accuracy_mult(kald8_out_ts)];

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
max1 = max(Mat_boxplot1);       % Taxa máxima de acerto
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
max2 = max(Mat_boxplot2);       % Taxa máxima de acerto
plot(media2,'*k')
hold off

%% SAVE DATA

save(OPT.file);

%% END