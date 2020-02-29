%% Machine Learning ToolBox

% K2NN - For sample selecting - MNIST
% Author: David Nascimento Coelho
% Last Update: 2019/11/29

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% GENERAL DEFINITIONS

% General options' structure

OPT.prob = 21;           	% Which problem will be solved / used
OPT.prob2 = 30;          	% More details about a specific data set
OPT.norm = 3;               % Normalization definition
OPT.lbl = 1;                % Labeling definition
OPT.Nr = 10;                % Number of repetitions of each algorithm
OPT.hold = 2;               % Hold out method
OPT.ptrn = 0.8;             % Percentage of samples for training
OPT.file = 'fileX.mat';     % file where all the variables will be saved

%% DATA LOADING AND PRE-PROCESSING

DATA = data_class_loading(OPT);     % Load Data Set
DATA = normalize(DATA,OPT);         % normalize the attributes' matrix
DATA = label_encode(DATA,OPT);      % adjust labels for the problem

%% HIPERPARAMETERS - DEFAULT

K2NNp.Dm = 2;                   % Design Method
K2NNp.Ss = 1;                   % Sparsification strategy
K2NNp.Ps = 1;                   % Prunning strategy
K2NNp.Us = 1;                   % Update strategy
K2NNp.v1 = 0.2;                 % Sparseness parameter 1
K2NNp.v2 = 0.9;                 % Sparseness parameter 2
K2NNp.K = 1;                    % number of nearest neighbors (classify)
K2NNp.Ktype = 2;                % kernel type ( see kernel_func() )
K2NNp.sig2n = 0.001;            % Kernel regularization parameter
K2NNp.sigma = 50;               % kernel hyperparameter ( see kernel_func()

LSSVCp.lambda = 50;             % Regularization Parameters
LSSVCp.Ktype = K2NNp.Ktype;  	% kernel hyperparameter ( see kernel_func()
LSSVCp.sig2n = K2NNp.sig2n;   	% Kernel regularization parameter
LSSVCp.sigma = K2NNp.sigma;  	% Variance

KQD1p.Ctype = 1;                % Type of classifier (Inverse Covariance)
KQD1p.Ktype = K2NNp.Ktype;  	% kernel hyperparameter ( see kernel_func()
KQD1p.sig2n = K2NNp.sig2n;    	% Kernel regularization parameter
KQD1p.sigma = K2NNp.sigma;  	% Variance

KQD2p.Ctype = 2;                % Type of classifier (Regularized Covariance)
KQD2p.Ktype = K2NNp.Ktype;  	% kernel hyperparameter ( see kernel_func()
KQD2p.sig2n = K2NNp.sig2n;    	% Kernel regularization parameter
KQD2p.sigma = K2NNp.sigma;  	% Variance

KRRp.Ktype = K2NNp.Ktype;       % kernel hyperparameter ( see kernel_func()
KRRp.sig2n = K2NNp.sig2n;       % Kernel regularization parameter
KRRp.sigma = K2NNp.sigma;       % Variance

%% ACCUMULATORS

% Data Accumulator

DATA_acc = cell(OPT.Nr,1);

% Names Accumulator

NAMES = {'K2NN','LSSVC','KQD-IC','KQD-RG','KRRC'};

% General statistics cell

nStats_tr_comp = cell(length(NAMES),1);
nStats_ts_comp = cell(length(NAMES),1);

% Classifiers Results and Parameters Accumulators

K2NNp_acc = cell(OPT.Nr,1);         % Acc of Parameters and Hyperparameters
K2NN_stats_tr = cell(OPT.Nr,1);     % Acc of Statistics of training data
K2NN_stats_ts = cell(OPT.Nr,1);     % Acc of Statistics of test data

LSSVCp_acc = cell(OPT.Nr,1);      	% Acc of Parameters and Hyperparameters
LSSVC_stats_tr = cell(OPT.Nr,1);	% Acc of Statistics of training data
LSSVC_stats_ts = cell(OPT.Nr,1);  	% Acc of Statistics of test data

KQD1p_acc = cell(OPT.Nr,1);      	% Acc of Parameters and Hyperparameters
KQD1_stats_tr = cell(OPT.Nr,1);     % Acc of Statistics of training data
KQD1_stats_ts = cell(OPT.Nr,1);  	% Acc of Statistics of test data

KQD2p_acc = cell(OPT.Nr,1);      	% Acc of Parameters and Hyperparameters
KQD2_stats_tr = cell(OPT.Nr,1);     % Acc of Statistics of training data
KQD2_stats_ts = cell(OPT.Nr,1);  	% Acc of Statistics of test data

KRRp_acc = cell(OPT.Nr,1);      	% Acc of Parameters and Hyperparameters
KRR_stats_tr = cell(OPT.Nr,1);      % Acc of Statistics of training data
KRR_stats_ts = cell(OPT.Nr,1);  	% Acc of Statistics of test data


%% HOLD OUT / TRAINING / TEST / STATISTICS

display('Begin Algorithm');

for r = 1:OPT.Nr,

% %%%%%%%%% DISPLAY REPETITION AND DURATION %%%%%%%%%%%%%%

display(r);
display(datestr(now));

% %%%%%%%%%%%%%%%%%%%% HOLD OUT %%%%%%%%%%%%%%%%%%%%%%%%%%

DATA_acc{r} = hold_out(DATA,OPT);   % Hold Out Function
DATAtr = DATA_acc{r}.DATAtr;        % Training Data
DATAts = DATA_acc{r}.DATAts;      	% Test Data

% %%%%%%%%%%%%%% SHUFFLE TRAINING DATA %%%%%%%%%%%%%%%%%%%

I = randperm(size(DATAtr.input,2));
DATAtr.input = DATAtr.input(:,I);
DATAtr.output = DATAtr.output(:,I);
DATAtr.lbl = DATAtr.lbl(:,I);

% %%%%%%%%%%%%%% CLASSIFIER'S TRAINING %%%%%%%%%%%%%%%%%%%

% Build dictionary (select prototypes)

K2NNp_acc{r}  = k2nn_train(DATAtr,K2NNp);

DATAtr_aux.input = K2NNp_acc{r}.Cx;
DATAtr_aux.output = K2NNp_acc{r}.Cy;

% Train other models just with selected prototypes

LSSVCp_acc{r} = lssvc_train(DATAtr_aux,LSSVCp);
KQD1p_acc{r} = kqd_train(DATAtr_aux,KQD1p);
KQD2p_acc{r} = kqd_train(DATAtr_aux,KQD2p);
KRRp_acc{r} = krr_train(DATAtr_aux,KRRp);

% %%%%%%%%% CLASSIFIER'S TEST AND STATISTICS %%%%%%%%%%%%%

OUTtr = k2nn_classify(DATAtr,K2NNp_acc{r});
K2NN_stats_tr{r} = class_stats_1turn(DATAtr,OUTtr);
OUTts = k2nn_classify(DATAts,K2NNp_acc{r});
K2NN_stats_ts{r} = class_stats_1turn(DATAts,OUTts);

OUTtr = lssvc_classify(DATAtr,LSSVCp_acc{r});
LSSVC_stats_tr{r} = class_stats_1turn(DATAtr,OUTtr);
OUTts = lssvc_classify(DATAts,LSSVCp_acc{r});
LSSVC_stats_ts{r} = class_stats_1turn(DATAts,OUTts);

OUTtr = kqd_classify(DATAtr,KQD1p_acc{r});
KQD1_stats_tr{r} = class_stats_1turn(DATAtr,OUTtr);
OUTts = kqd_classify(DATAts,KQD1p_acc{r});
KQD1_stats_ts{r} = class_stats_1turn(DATAts,OUTts);

OUTtr = kqd_classify(DATAtr,KQD2p_acc{r});
KQD2_stats_tr{r} = class_stats_1turn(DATAtr,OUTtr);
OUTts = kqd_classify(DATAts,KQD2p_acc{r});
KQD2_stats_ts{r} = class_stats_1turn(DATAts,OUTts);

OUTtr = krr_classify(DATAtr,KRRp_acc{r});
KRR_stats_tr{r} = class_stats_1turn(DATAtr,OUTtr);
OUTts = krr_classify(DATAts,KRRp_acc{r});
KRR_stats_ts{r} = class_stats_1turn(DATAts,OUTts);

end

display('Finish Algorithm')
display(datestr(now));

%% STATISTICS FOR N TURNS

% Training Data

nStats_tr_k2nn = class_stats_nturns(K2NN_stats_tr);
nStats_tr_lssvc = class_stats_nturns(LSSVC_stats_tr);
nStats_tr_kqd1 = class_stats_nturns(KQD1_stats_tr);
nStats_tr_kqd2 = class_stats_nturns(KQD2_stats_tr);
nStats_tr_krr = class_stats_nturns(KRR_stats_tr);

nStats_tr_comp{1,1} = nStats_tr_k2nn;
nStats_tr_comp{2,1} = nStats_tr_lssvc;
nStats_tr_comp{3,1} = nStats_tr_kqd1;
nStats_tr_comp{4,1} = nStats_tr_kqd2;
nStats_tr_comp{5,1} = nStats_tr_krr;

% Test Data

nStats_ts_k2nn = class_stats_nturns(K2NN_stats_ts);
nStats_ts_lssvc = class_stats_nturns(LSSVC_stats_ts);
nStats_ts_kqd1 = class_stats_nturns(KQD1_stats_ts);
nStats_ts_kqd2 = class_stats_nturns(KQD2_stats_ts);
nStats_ts_krr = class_stats_nturns(KRR_stats_ts);

nStats_ts_comp{1,1} = nStats_ts_k2nn;
nStats_ts_comp{2,1} = nStats_ts_lssvc;
nStats_ts_comp{3,1} = nStats_ts_kqd1;
nStats_ts_comp{4,1} = nStats_ts_kqd2;
nStats_ts_comp{5,1} = nStats_ts_krr;

%% BOXPLOT OF TEST ACCURACIES

% Traning Data

class_stats_ncomp(nStats_tr_comp,NAMES);

% Test Data

class_stats_ncomp(nStats_ts_comp,NAMES);

%% SAVE DATA

% save(OPT.file);

%% END