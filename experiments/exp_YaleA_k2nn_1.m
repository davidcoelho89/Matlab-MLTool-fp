%% Machine Learning ToolBox

% Test For Yale A Images DataBase and K2NN classifier
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
OPT.Nr = 10;                % Number of repetitions of the algorithm
OPT.hold = 02;              % Hold out method
OPT.ptrn = 0.70;            % Percentage of samples for training
OPT.file = 'k2nn_tst4.mat'; % file where all the variables will be saved

% Cross Validation hiperparameters

CVp.on = 1;                 % If 1, includes cross validation
CVp.fold = 5;               % Number of folds for cross validation
CVp.lambda = 0.5;         	% weight of number of prototypes

%% DATA LOADING AND PRE-PROCESSING

DATA = data_class_loading(OPT);     % Load Data Set
DATA = normalize(DATA,OPT);         % normalize the attributes' matrix
DATA = label_encode(DATA,OPT);      % adjust labels for the problem

%% HYPERPARAMETERS - DEFAULT

% Ktype = Linear, Dm = 1 (one dictionary for all dataset)

K2NN1p_acc = cell(OPT.Nr,1);    % Init of Acc Hyperparameters of K2NN-1
PAR_k2nn_1.Ss = 01;             % Sparsification strategy
PAR_k2nn_1.Us = 01;             % Update strategy
PAR_k2nn_1.Ps = 01;             % Prunning strategy
PAR_k2nn_1.Dm = 01;             % Design Method
PAR_k2nn_1.v1 = 0.9;           	% Sparseness
PAR_k2nn_1.Ktype = 01;          % Kernel Type
PAR_k2nn_1.sigma = 02;          % Comprehensiveness of kernel

% Ktype = Linear, Dm = 2 (one dictionary for each class)

K2NN2p_acc = cell(OPT.Nr,1);    % Init of Acc Hyperparameters of K2NN-2
PAR_k2nn_2.Ss = 01;             % Sparsification strategy (ALD, Coherence...)
PAR_k2nn_2.Us = 01;             % Update strategy (LMS, RLS...)
PAR_k2nn_2.Ps = 01;             % Prunning strategy (penalization...)
PAR_k2nn_2.Dm = 02;             % Design Method (per class or per dataset)
PAR_k2nn_2.v1 = 0.9;          	% Sparseness
PAR_k2nn_2.Ktype = 01;          % Kernel Type
PAR_k2nn_2.sigma = 02;          % Comprehensiveness of kernel

% Ktype = Gaussian, Dm = 1 (one dictionary for all dataset)

K2NN3p_acc = cell(OPT.Nr,1);    % Init of Acc Hyperparameters of K2NN-1
PAR_k2nn_3.Ss = 01;             % Sparsification strategy
PAR_k2nn_3.Us = 01;             % Update strategy
PAR_k2nn_3.Ps = 01;             % Prunning strategy
PAR_k2nn_3.Dm = 01;             % Design Method
PAR_k2nn_3.v1 = 0.9;           	% Sparseness
PAR_k2nn_3.Ktype = 02;          % Kernel Type
PAR_k2nn_3.sigma = 02;          % Comprehensiveness of kernel

% Ktype = Gaussian, Dm = 2 (one dictionary for each class)

K2NN4p_acc = cell(OPT.Nr,1);    % Init of Acc Hyperparameters of K2NN-1
PAR_k2nn_4.Ss = 01;             % Sparsification strategy
PAR_k2nn_4.Us = 01;             % Update strategy
PAR_k2nn_4.Ps = 01;             % Prunning strategy
PAR_k2nn_4.Dm = 02;             % Design Method
PAR_k2nn_4.v1 = 0.9;        	% Sparseness
PAR_k2nn_4.Ktype = 02;          % Kernel Type
PAR_k2nn_4.sigma = 02;          % Comprehensiveness of kernel

% Ktype = Cauchy, Dm = 1 (one dictionary for all dataset)

K2NN5p_acc = cell(OPT.Nr,1);    % Init of Acc Hyperparameters of K2NN-1
PAR_k2nn_5.Ss = 01;             % Sparsification strategy
PAR_k2nn_5.Us = 01;             % Update strategy
PAR_k2nn_5.Ps = 01;             % Prunning strategy
PAR_k2nn_5.Dm = 01;             % Design Method
PAR_k2nn_5.v1 = 0.9;            % Sparseness
PAR_k2nn_5.Ktype = 05;          % Kernel Type
PAR_k2nn_5.sigma = 02;          % Comprehensiveness of kernel

% Ktype = Cauchy, Dm = 2 (one dictionary for each class)

K2NN6p_acc = cell(OPT.Nr,1);    % Init of Acc Hyperparameters of K2NN-1
PAR_k2nn_6.Ss = 01;             % Sparsification strategy
PAR_k2nn_6.Us = 01;             % Update strategy
PAR_k2nn_6.Ps = 01;             % Prunning strategy
PAR_k2nn_6.Dm = 02;             % Design Method
PAR_k2nn_6.v1 = 0.9;            % Sparseness
PAR_k2nn_6.Ktype = 05;          % Kernel Type
PAR_k2nn_6.sigma = 02;          % Comprehensiveness of kernel

% Ktype = Log, Dm = 1 (one dictionary for all dataset)

K2NN7p_acc = cell(OPT.Nr,1);    % Init of Acc Hyperparameters of K2NN-1
PAR_k2nn_7.Ss = 01;             % Sparsification strategy
PAR_k2nn_7.Us = 01;             % Update strategy
PAR_k2nn_7.Ps = 01;             % Prunning strategy
PAR_k2nn_7.Dm = 01;             % Design Method
PAR_k2nn_7.v1 = 0.9;            % Sparseness
PAR_k2nn_7.Ktype = 06;          % Kernel Type
PAR_k2nn_7.sigma = 02;          % Comprehensiveness of kernel

% Ktype = Log, Dm = 2 (one dictionary for each class)

K2NN8p_acc = cell(OPT.Nr,1);    % Init of Acc Hyperparameters of K2NN-1
PAR_k2nn_8.Ss = 01;             % Sparsification strategy
PAR_k2nn_8.Us = 01;             % Update strategy
PAR_k2nn_8.Ps = 01;             % Prunning strategy
PAR_k2nn_8.Dm = 02;             % Design Method
PAR_k2nn_8.v1 = 0.9;            % Sparseness
PAR_k2nn_8.Ktype = 06;          % Kernel Type
PAR_k2nn_8.sigma = 02;          % Comprehensiveness of kernel

%% HIPERPARAMETERS - GRID FOR CROSS VALIDATION

% Ss = 1 / Us = ? / Ps = ? / Dm = 1,2 / v = CV / Ktype = 1,2,5,6 / sig2 = CV

% Get Default Hyperparameters

K2NN1cv = PAR_k2nn_1;
K2NN2cv = PAR_k2nn_2;
K2NN3cv = PAR_k2nn_3;
K2NN4cv = PAR_k2nn_4;
K2NN5cv = PAR_k2nn_5;
K2NN6cv = PAR_k2nn_6;
K2NN7cv = PAR_k2nn_7;
K2NN8cv = PAR_k2nn_8;

% Set Variable HyperParameters

if CVp.on == 1

K2NN1cv.v1 = 2.^linspace(-4,3,8);

K2NN2cv.v1 = 2.^linspace(-4,3,8);

K2NN3cv.v1 = 2.^linspace(-4,3,8);
K2NN3cv.sigma = 2.^linspace(-10,9,20);

K2NN4cv.v1 = 2.^linspace(-4,3,8);
K2NN4cv.sigma = 2.^linspace(-10,9,20);

K2NN5cv.v1 = 2.^linspace(-4,3,8);
K2NN5cv.sigma = 2.^linspace(-10,9,20);

K2NN6cv.v1 = 2.^linspace(-4,3,8);
K2NN6cv.sigma = 2.^linspace(-10,9,20);

K2NN7cv.v1 = -2.^linspace(10,2,9);
K2NN7cv.sigma = 2.^linspace(-10,9,20);

K2NN8cv.v1 = -2.^linspace(10,2,9);
K2NN8cv.sigma = 2.^linspace(-10,9,20);

end

%% ACCUMULATORS

% Acc labels and data division

hold_acc = cell(OPT.Nr,1);

% Acc statistics from train and test samples

K2NN1_tr_acc = cell(OPT.Nr,1);   	% Acc of Statistics of training data
K2NN1_ts_acc = cell(OPT.Nr,1);   	% Acc of Statistics of test data

K2NN2_tr_acc = cell(OPT.Nr,1);   	% Acc of Statistics of training data
K2NN2_ts_acc = cell(OPT.Nr,1);   	% Acc of Statistics of test data

K2NN3_tr_acc = cell(OPT.Nr,1);   	% Acc of Statistics of training data
K2NN3_ts_acc = cell(OPT.Nr,1);   	% Acc of Statistics of test data

K2NN4_tr_acc = cell(OPT.Nr,1);   	% Acc of Statistics of training data
K2NN4_ts_acc = cell(OPT.Nr,1);   	% Acc of Statistics of test data

K2NN5_tr_acc = cell(OPT.Nr,1);   	% Acc of Statistics of training data
K2NN5_ts_acc = cell(OPT.Nr,1);   	% Acc of Statistics of test data

K2NN6_tr_acc = cell(OPT.Nr,1);   	% Acc of Statistics of training data
K2NN6_ts_acc = cell(OPT.Nr,1);   	% Acc of Statistics of test data

K2NN7_tr_acc = cell(OPT.Nr,1);   	% Acc of Statistics of training data
K2NN7_ts_acc = cell(OPT.Nr,1);   	% Acc of Statistics of test data

K2NN8_tr_acc = cell(OPT.Nr,1);   	% Acc of Statistics of training data
K2NN8_ts_acc = cell(OPT.Nr,1);   	% Acc of Statistics of test data


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

disp('He 1')

[PAR_k2nn_1] = cross_valid_gs2(DATAtr,CVp,K2NN1cv,@k2nn_train,@k2nn_classify);

disp('He 2')

[PAR_k2nn_2] = cross_valid_gs2(DATAtr,CVp,K2NN2cv,@k2nn_train,@k2nn_classify);

[PAR_k2nn_3] = cross_valid_gs2(DATAtr,CVp,K2NN3cv,@k2nn_train,@k2nn_classify);

[PAR_k2nn_4] = cross_valid_gs2(DATAtr,CVp,K2NN4cv,@k2nn_train,@k2nn_classify);

[PAR_k2nn_5] = cross_valid_gs2(DATAtr,CVp,K2NN5cv,@k2nn_train,@k2nn_classify);

[PAR_k2nn_6] = cross_valid_gs2(DATAtr,CVp,K2NN6cv,@k2nn_train,@k2nn_classify);

[PAR_k2nn_7] = cross_valid_gs2(DATAtr,CVp,K2NN7cv,@k2nn_train,@k2nn_classify);

[PAR_k2nn_8] = cross_valid_gs2(DATAtr,CVp,K2NN8cv,@k2nn_train,@k2nn_classify);

% %%%%%%%%%%%%%% CLASSIFIER'S TRAINING %%%%%%%%%%%%%%%%%%%

[PAR_k2nn_1] = k2nn_train(DATAtr,PAR_k2nn_1);	% Calculate parameters
K2NN1p_acc{r} = PAR_k2nn_1;                  	% Acc Parameters

[PAR_k2nn_2] = k2nn_train(DATAtr,PAR_k2nn_2);	% Calculate parameters
K2NN2p_acc{r} = PAR_k2nn_2;                  	% Acc Parameters

[PAR_k2nn_3] = k2nn_train(DATAtr,PAR_k2nn_3);	% Calculate parameters
K2NN3p_acc{r} = PAR_k2nn_3;                  	% Acc Parameters

[PAR_k2nn_4] = k2nn_train(DATAtr,PAR_k2nn_4);	% Calculate parameters
K2NN4p_acc{r} = PAR_k2nn_4;                  	% Acc Parameters

[PAR_k2nn_5] = k2nn_train(DATAtr,PAR_k2nn_5);	% Calculate parameters
K2NN5p_acc{r} = PAR_k2nn_5;                  	% Acc Parameters

[PAR_k2nn_6] = k2nn_train(DATAtr,PAR_k2nn_6);	% Calculate parameters
K2NN6p_acc{r} = PAR_k2nn_6;                  	% Acc Parameters

[PAR_k2nn_7] = k2nn_train(DATAtr,PAR_k2nn_7);	% Calculate parameters
K2NN7p_acc{r} = PAR_k2nn_7;                  	% Acc Parameters

[PAR_k2nn_8] = k2nn_train(DATAtr,PAR_k2nn_8);	% Calculate parameters
K2NN8p_acc{r} = PAR_k2nn_8;                  	% Acc Parameters

% %%%%%%%%% CLASSIFIER'S TEST AND STATISTICS %%%%%%%%%%%%%

% Ktype = Linear, Dm = 1 (one dictionary for all dataset)

[OUTtr] = k2nn_classify(DATAtr,PAR_k2nn_1);   	% Outputs with training data
[STATS_tr] = class_stats_1turn(DATAtr,OUTtr);   % Results with training data
K2NN1_tr_acc{r} = STATS_tr;                     % Acc Training Statistics

[OUTts] = k2nn_classify(DATAts,PAR_k2nn_1); 	% Outputs with test data
[STATS_ts] = class_stats_1turn(DATAts,OUTts);   % Results with test data
K2NN1_ts_acc{r} = STATS_ts;                     % Acc Test Statistics

% Ktype = Linear, Dm = 2 (one dictionary for each class)

[OUTtr] = k2nn_classify(DATAtr,PAR_k2nn_2);   	% Outputs with training data
[STATS_tr] = class_stats_1turn(DATAtr,OUTtr);   % Results with training data
K2NN2_tr_acc{r} = STATS_tr;                     % Acc Training Statistics

[OUTts] = k2nn_classify(DATAts,PAR_k2nn_2); 	% Outputs with test data
[STATS_ts] = class_stats_1turn(DATAts,OUTts);   % Results with test data
K2NN2_ts_acc{r} = STATS_ts;                     % Acc Test Statistics

% Ktype = Gaussian, Dm = 1 (one dictionary for all dataset)

[OUTtr] = k2nn_classify(DATAtr,PAR_k2nn_3);   	% Outputs with training data
[STATS_tr] = class_stats_1turn(DATAtr,OUTtr);   % Results with training data
K2NN3_tr_acc{r} = STATS_tr;                     % Acc Training Statistics

[OUTts] = k2nn_classify(DATAts,PAR_k2nn_3); 	% Outputs with test data
[STATS_ts] = class_stats_1turn(DATAts,OUTts);   % Results with test data
K2NN3_ts_acc{r} = STATS_ts;                     % Acc Test Statistics

% Ktype = Gaussian, Dm = 2 (one dictionary for each class)

[OUTtr] = k2nn_classify(DATAtr,PAR_k2nn_4);   	% Outputs with training data
[STATS_tr] = class_stats_1turn(DATAtr,OUTtr);   % Results with training data
K2NN4_tr_acc{r} = STATS_tr;                     % Acc Training Statistics

[OUTts] = k2nn_classify(DATAts,PAR_k2nn_4); 	% Outputs with test data
[STATS_ts] = class_stats_1turn(DATAts,OUTts);   % Results with test data
K2NN4_ts_acc{r} = STATS_ts;                     % Acc Test Statistics

% Ktype = Cauchy, Dm = 1 (one dictionary for all dataset)

[OUTtr] = k2nn_classify(DATAtr,PAR_k2nn_5);   	% Outputs with training data
[STATS_tr] = class_stats_1turn(DATAtr,OUTtr);   % Results with training data
K2NN5_tr_acc{r} = STATS_tr;                     % Acc Training Statistics

[OUTts] = k2nn_classify(DATAts,PAR_k2nn_5); 	% Outputs with test data
[STATS_ts] = class_stats_1turn(DATAts,OUTts);   % Results with test data
K2NN5_ts_acc{r} = STATS_ts;                     % Acc Test Statistics

% Ktype = Cauchy, Dm = 2 (one dictionary for each class)

[OUTtr] = k2nn_classify(DATAtr,PAR_k2nn_6);   	% Outputs with training data
[STATS_tr] = class_stats_1turn(DATAtr,OUTtr);   % Results with training data
K2NN6_tr_acc{r} = STATS_tr;                     % Acc Training Statistics

[OUTts] = k2nn_classify(DATAts,PAR_k2nn_6); 	% Outputs with test data
[STATS_ts] = class_stats_1turn(DATAts,OUTts);   % Results with test data
K2NN6_ts_acc{r} = STATS_ts;                     % Acc Test Statistics

% Ktype = Log, Dm = 1 (one dictionary for all dataset)

[OUTtr] = k2nn_classify(DATAtr,PAR_k2nn_7);   	% Outputs with training data
[STATS_tr] = class_stats_1turn(DATAtr,OUTtr);   % Results with training data
K2NN7_tr_acc{r} = STATS_tr;                     % Acc Training Statistics

[OUTts] = k2nn_classify(DATAts,PAR_k2nn_7); 	% Outputs with test data
[STATS_ts] = class_stats_1turn(DATAts,OUTts);   % Results with test data
K2NN7_ts_acc{r} = STATS_ts;                     % Acc Test Statistics

% Ktype = Log, Dm = 2 (one dictionary for each class)

[OUTtr] = k2nn_classify(DATAtr,PAR_k2nn_8);   	% Outputs with training data
[STATS_tr] = class_stats_1turn(DATAtr,OUTtr);   % Results with training data
K2NN8_tr_acc{r} = STATS_tr;                     % Acc Training Statistics

[OUTts] = k2nn_classify(DATAts,PAR_k2nn_8); 	% Outputs with test data
[STATS_ts] = class_stats_1turn(DATAts,OUTts);   % Results with test data
K2NN8_ts_acc{r} = STATS_ts;                     % Acc Test Statistics

end

disp('Finish Algorithm')
display(datestr(now));

%% RESULTS / STATISTICS / GRAPHICS

% Statistics for n turns

nSTATS_tr_1 = class_stats_nturns(K2NN1_tr_acc);
nSTATS_ts_1 = class_stats_nturns(K2NN1_ts_acc);

nSTATS_tr_2 = class_stats_nturns(K2NN2_tr_acc);
nSTATS_ts_2 = class_stats_nturns(K2NN2_ts_acc);

nSTATS_tr_3 = class_stats_nturns(K2NN3_tr_acc);
nSTATS_ts_3 = class_stats_nturns(K2NN3_ts_acc);

nSTATS_tr_4 = class_stats_nturns(K2NN4_tr_acc);
nSTATS_ts_4 = class_stats_nturns(K2NN4_ts_acc);

nSTATS_tr_5 = class_stats_nturns(K2NN5_tr_acc);
nSTATS_ts_5 = class_stats_nturns(K2NN5_ts_acc);

nSTATS_tr_6 = class_stats_nturns(K2NN6_tr_acc);
nSTATS_ts_6 = class_stats_nturns(K2NN6_ts_acc);

nSTATS_tr_7 = class_stats_nturns(K2NN7_tr_acc);
nSTATS_ts_7 = class_stats_nturns(K2NN7_ts_acc);

nSTATS_tr_8 = class_stats_nturns(K2NN8_tr_acc);
nSTATS_ts_8 = class_stats_nturns(K2NN8_ts_acc);

% Boxplots Init

labels = {};

Mat_boxplot1 = []; % Train Multiclass
Mat_boxplot2 = []; % Test Multiclass

% K2NN-1

[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'K2NN-1'};
Mat_boxplot1 = [Mat_boxplot1 nSTATS_tr_1.acc'];
Mat_boxplot2 = [Mat_boxplot2 nSTATS_ts_1.acc'];

% K2NN-2

[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'K2NN-2'};
Mat_boxplot1 = [Mat_boxplot1 nSTATS_tr_2.acc'];
Mat_boxplot2 = [Mat_boxplot2 nSTATS_ts_2.acc'];

% K2NN-3

[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'K2NN-G1'};
Mat_boxplot1 = [Mat_boxplot1 nSTATS_tr_3.acc'];
Mat_boxplot2 = [Mat_boxplot2 nSTATS_ts_3.acc'];

% K2NN-4

[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'K2NN-G2'};
Mat_boxplot1 = [Mat_boxplot1 nSTATS_tr_4.acc'];
Mat_boxplot2 = [Mat_boxplot2 nSTATS_ts_4.acc'];

% K2NN-5

[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'K2NN-C1'};
Mat_boxplot1 = [Mat_boxplot1 nSTATS_tr_5.acc'];
Mat_boxplot2 = [Mat_boxplot2 nSTATS_ts_5.acc'];

% K2NN-6

[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'K2NN-C2'};
Mat_boxplot1 = [Mat_boxplot1 nSTATS_tr_6.acc'];
Mat_boxplot2 = [Mat_boxplot2 nSTATS_ts_6.acc'];

% K2NN-7

[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'K2NN-L1'};
Mat_boxplot1 = [Mat_boxplot1 nSTATS_tr_7.acc'];
Mat_boxplot2 = [Mat_boxplot2 nSTATS_ts_7.acc'];

% K2NN-8

[~,n_labels] = size(labels);
n_labels = n_labels+1;
labels(1,n_labels) = {'K2NN-L2'};
Mat_boxplot1 = [Mat_boxplot1 nSTATS_tr_8.acc'];
Mat_boxplot2 = [Mat_boxplot2 nSTATS_ts_8.acc'];

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

save(OPT.file);

%% END