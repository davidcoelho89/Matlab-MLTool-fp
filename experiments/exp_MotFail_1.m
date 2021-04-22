%% Machine Learning ToolBox

% Motor Failure Test - 7 classifiers, with reject option
% Author: David Nascimento Coelho
% Last Update: 2019/10/01

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% GENERAL DEFINITIONS

% General options' structure

OPT.prob = 7;               % Which problem will be solved / used
OPT.prob2 = 2;              % More details about a specific data set
OPT.norm = 3;               % Normalization definition
OPT.lbl = 1;                % Labeling definition
OPT.Nr = 50;                % Number of repetitions of each algorithm
OPT.hold = 2;               % Hold out method
OPT.ptrn = 0.8;             % Percentage of samples for training
OPT.file = 'motor_fail_2.mat'; % file where all the variables will be saved

% Cross Validation hiperparameters

CVp.fold = 5;               % Number of folds for cross validation

% Reject Option hiperparameters

REJp.band = 0.3;            % Range of rejected values [-band +band]
REJp.w = 0.25;              % Rejection cost

%% DATA LOADING AND PRE-PROCESSING

DATA = data_class_loading(OPT);     % Load Data Set
DATA = normalize(DATA,OPT);         % normalize the attributes' matrix
DATA = label_encode(DATA,OPT);      % adjust labels for the problem

%% HIPERPARAMETERS - DEFAULT

% If an specific hyperparameter is not set, the algorithm uses a default value.

OLShp.aprox = 1;          	% Type of aproximation

BAYhp.type = 2;           	% Type of classificer

PShp.Ne = 200;             	% maximum number of training epochs
PShp.eta = 0.05;          	% Learning step

MLPhp.Nh = 10;            	% Number of hidden neurons
MLPhp.Ne = 200;           	% Maximum training epochs
MLPhp.eta = 0.05;         	% Learning Step
MLPhp.mom = 0.75;         	% Moment Factor
MLPhp.Nlin = 2;          	% Non-linearity

ELMhp.Nh = 25;            	% Number of hidden neurons
ELMhp.Nlin = 2;           	% Non-linearity

SVChp.lambda = 5;         	% Regularization Constant
SVChp.Ktype = 2;         	% Kernel Type
SVChp.sigma = 0.01;       	% Variance

LSSVChp.lambda = 0.5;     	% Regularization Constant
LSSVChp.Ktype = 2;       	% Kernel Type
LSSVChp.sigma = 128;    	% Variance

MLMhp.K = 09;           	% Number of reference points

%% HIPERPARAMETERS - GRID FOR CROSS VALIDATION

if CVp.on == 1
    
    OLScv.aprox = 1;
    
    BAYcv.type = 2;
    
    PScv.Ne = 200;
    PScv.eta = 0.05;
    
    MLPcv.Nh = 2:20;
    MLPcv.Ne = 200;
    MLPcv.eta = 0.05;
    MLPcv.mom = 0.75;
    MLPcv.Nlin = 2;
    
    ELMcv.Nh = 10:30;
    ELMcv.Nlin = 2;
    
    SVCcv.lambda = [0.5 5 10 15 25 50 100 250 500 1000];
    SVCcv.Ktype = 2;
    SVCcv.sigma = [0.01 0.05 0.1 0.5 1 5 10 50 100 500];
    
    LSSVCcv.lambda = 2.^linspace(-5,20,26);
    LSSVCcv.Ktype = 2;
    LSSVCcv.sigma = 2.^linspace(-10,10,21);
    
    MLMcv.K = 2:15;
   
end

%% ACCUMULATORS

% Acc of names for plots

NAMES = {'OLS','BAYES','PS','MLP','ELM','SVC','LSSVC','MLM'};

% Acc of labels and data division

DATA_acc = cell(OPT.Nr,1);

% General statistics cell

nSTATS_all_tr = cell(8,1);
nSTATS_all_ts = cell(8,1);

% Acc of outputs and statistics for each classifier

OLSp_acc = cell(OPT.Nr,1);          % Acc Hyperparameters of OLS
ols_out_tr = cell(OPT.Nr,1);        % Acc of training data output
ols_stats_tr = cell(OPT.Nr,1);      % Acc of statistics from training
ols_out_ts = cell(OPT.Nr,1);        % Acc of test data output
ols_stats_ts = cell(OPT.Nr,1);      % Acc of statistics from test

BAYp_acc = cell(OPT.Nr,1);          % Acc Hyperparameters of Bayes
bay_out_tr = cell(OPT.Nr,1);        % Acc of training data output
bay_stats_tr = cell(OPT.Nr,1);      % Acc of statistics from training
bay_out_ts = cell(OPT.Nr,1);        % Acc of test data output
bay_stats_ts = cell(OPT.Nr,1);      % Acc of statistics from test

PSp_acc = cell(OPT.Nr,1);           % Acc Hyperparameters of PS
ps_out_tr = cell(OPT.Nr,1);         % Acc of training data output
ps_stats_tr = cell(OPT.Nr,1);       % Acc of statistics from training
ps_out_ts = cell(OPT.Nr,1);         % Acc of test data output
ps_stats_ts = cell(OPT.Nr,1);       % Acc of statistics from test

MLPp_acc = cell(OPT.Nr,1);          % Acc Hyperparameters of MLP
mlp_out_tr = cell(OPT.Nr,1);        % Acc of training data output
mlp_stats_tr = cell(OPT.Nr,1);      % Acc of statistics from training
mlp_out_ts = cell(OPT.Nr,1);        % Acc of test data output
mlp_stats_ts = cell(OPT.Nr,1);  	% Acc of statistics from test

ELMp_acc = cell(OPT.Nr,1);          % Acc Hyperparameters of ELM
elm_out_tr = cell(OPT.Nr,1);        % Acc of training data output
elm_stats_tr = cell(OPT.Nr,1);      % Acc of statistics from training
elm_out_ts = cell(OPT.Nr,1);        % Acc of test data output
elm_stats_ts = cell(OPT.Nr,1);      % Acc of statistics from test

SVCp_acc = cell(OPT.Nr,1);          % Acc Hyperparameters of SVM
svc_out_tr = cell(OPT.Nr,1);        % Acc of training data output
svc_stats_tr = cell(OPT.Nr,1);      % Acc of statistics from training
svc_out_ts = cell(OPT.Nr,1);        % Acc of test data output
svc_stats_ts = cell(OPT.Nr,1);      % Acc of statistics from test

LSSVCp_acc = cell(OPT.Nr,1);        % Acc Hyperparameters of LSSVM
lssvc_out_tr = cell(OPT.Nr,1);      % Acc of training data output
lssvc_stats_tr = cell(OPT.Nr,1);	% Acc of statistics from training
lssvc_out_ts = cell(OPT.Nr,1);      % Acc of test data output
lssvc_stats_ts = cell(OPT.Nr,1);	% Acc of statistics from test

MLMp_acc = cell(OPT.Nr,1);          % Acc Hyperparameters of MLM
mlm_out_tr = cell(OPT.Nr,1);        % Acc of training data output
mlm_stats_tr = cell(OPT.Nr,1);      % Acc of statistics from training
mlm_out_ts = cell(OPT.Nr,1);        % Acc of test data output
mlm_stats_ts = cell(OPT.Nr,1);      % Acc of statistics from test

%% HOLD OUT / CROSS VALIDATION / TRAINING / TEST

for r = 1:OPT.Nr

% %%%%%%%%% DISPLAY REPETITION AND DURATION %%%%%%%%%%%%%%

display(r);
display(datestr(now));

% %%%%%%%%%%%%%%%%%%%% HOLD OUT %%%%%%%%%%%%%%%%%%%%%%%%%%

DATA_acc{r} = hold_out(DATA,OPT);   % Hold Out Function
DATAtr = DATA_acc{r}.DATAtr;        % Training Data
DATAts = DATA_acc{r}.DATAts;      	% Test Data

% %%%%%%%%%%%%%%%% CROSS VALIDATION %%%%%%%%%%%%%%%%%%%%%

% With grid search method

OLShp   = cross_valid_gs(DATAtr,CVp,OLScv,@ols_train,@ols_classify);
BAYhp   = cross_valid_gs(DATAtr,CVp,BAYcv,@gauss_train,@gauss_classify);
PShp    = cross_valid_gs(DATAtr,CVp,PScv,@ps_train,@ps_classify);
MLPhp   = cross_valid_gs(DATAtr,CVp,MLPcv,@mlp_train,@mlp_classify);
ELMhp   = cross_valid_gs(DATAtr,CVp,ELMcv,@elm_train,@elm_classify);
SVChp   = cross_valid_gs(DATAtr,CVp,SVCcv,@svc_train,@svc_classify);
LSSVChp = cross_valid_gs(DATAtr,CVp,LSSVCcv,@lssvc_train,@lssvc_classify);
MLMhp   = cross_valid_gs(DATAtr,CVp,MLMcv,@mlm_train,@mlm_classify);

% %%%%%%%%%%%%%% CLASSIFIERS' TRAINING %%%%%%%%%%%%%%%%%%%

OLSp_acc{r}   = ols_train(DATAtr,OLShp);
BAYp_acc{r}   = gauss_train(DATAtr,BAYhp);
PSp_acc{r}    = ps_train(DATAtr,PShp);
MLPp_acc{r}   = mlp_train(DATAtr,MLPhp);
ELMp_acc{r}   = elm_train(DATAtr,ELMhp);
SVCp_acc{r}   = svc_train(DATAtr,SVChp);
LSSVCp_acc{r} = lssvc_train(DATAtr,LSSVChp);
MLMp_acc{r}   = mlm_train(DATAtr,MLMhp);

% %%%%%%%%%%%%%%%%% CLASSIFIERS' TEST %%%%%%%%%%%%%%%%%%%%

ols_out_tr{r} = ols_classify(DATAtr,OLSp_acc{r});
ols_stats_tr{r} = class_stats_1turn(DATAtr,ols_out_tr{r});
ols_out_ts{r} = ols_classify(DATAts,OLSp_acc{r});
ols_stats_ts{r} = class_stats_1turn(DATAts,ols_out_ts{r});

bay_out_tr{r} = gauss_classify(DATAtr,BAYp_acc{r});
bay_stats_tr{r} = class_stats_1turn(DATAtr,bay_out_tr{r});
bay_out_ts{r} = gauss_classify(DATAts,BAYp_acc{r});
bay_stats_ts{r} = class_stats_1turn(DATAts,bay_out_ts{r});

ps_out_tr{r} = ps_classify(DATAtr,PSp_acc{r});
ps_stats_tr{r} = class_stats_1turn(DATAtr,ps_out_tr{r});
ps_out_ts{r} = ps_classify(DATAts,PSp_acc{r});
ps_stats_ts{r} = class_stats_1turn(DATAts,ps_out_ts{r});

mlp_out_tr{r} = mlp_classify(DATAtr,MLPp_acc{r});
mlp_stats_tr{r} = class_stats_1turn(DATAtr,mlp_out_tr{r});
mlp_out_ts{r} = mlp_classify(DATAts,MLPp_acc{r});
mlp_stats_ts{r} = class_stats_1turn(DATAts,mlp_out_ts{r});

elm_out_tr{r} = elm_classify(DATAtr,ELMp_acc{r});
elm_stats_tr{r} = class_stats_1turn(DATAtr,elm_out_tr{r});
elm_out_ts{r} = elm_classify(DATAts,ELMp_acc{r});
elm_stats_ts{r} = class_stats_1turn(DATAts,elm_out_ts{r});

svc_out_tr{r} = svc_classify(DATAtr,SVCp_acc{r});
svc_stats_tr{r} = class_stats_1turn(DATAtr,svc_out_tr{r});
svc_out_ts{r} = svc_classify(DATAts,SVCp_acc{r});
svc_stats_ts{r} = class_stats_1turn(DATAts,svc_out_ts{r});

lssvc_out_tr{r} = lssvc_classify(DATAtr,LSSVCp_acc{r});
lssvc_stats_tr{r} = class_stats_1turn(DATAtr,lssvc_out_tr{r});
lssvc_out_ts{r} = lssvc_classify(DATAts,LSSVCp_acc{r});
lssvc_stats_ts{r} = class_stats_1turn(DATAts,lssvc_out_ts{r});

mlm_out_tr{r} = mlm_classify(DATAtr,MLMp_acc{r});
mlm_stats_tr{r} = class_stats_1turn(DATAtr,mlm_out_tr{r});
mlm_out_ts{r} = mlm_classify(DATAts,MLMp_acc{r});
mlm_stats_ts{r} = class_stats_1turn(DATAts,mlm_out_ts{r});

end

%% STATISTICS

% Statistics for n turns

nStats_tr_ols = class_stats_nturns(ols_stats_tr);
nStats_ts_ols = class_stats_nturns(ols_stats_ts);

nStats_tr_bay = class_stats_nturns(bay_stats_tr);
nStats_ts_bay = class_stats_nturns(bay_stats_ts);

nStats_tr_ps = class_stats_nturns(ps_stats_tr);
nStats_ts_ps = class_stats_nturns(ps_stats_ts);

nStats_tr_mlp = class_stats_nturns(mlp_stats_tr);
nStats_ts_mlp = class_stats_nturns(mlp_stats_ts);

nStats_tr_elm = class_stats_nturns(elm_stats_tr);
nStats_ts_elm = class_stats_nturns(elm_stats_ts);

nStats_tr_svc = class_stats_nturns(svc_stats_tr);
nStats_ts_svc = class_stats_nturns(svc_stats_ts);

nStats_tr_lssvc = class_stats_nturns(lssvc_stats_tr);
nStats_ts_lssvc = class_stats_nturns(lssvc_stats_ts);

nStats_tr_mlm = class_stats_nturns(mlm_stats_tr);
nStats_ts_mlm = class_stats_nturns(mlm_stats_ts);

% Get all Statistics in one Cell

nSTATS_all_tr{1,1} = nStats_tr_ols;
nSTATS_all_tr{2,1} = nStats_tr_bay;
nSTATS_all_tr{3,1} = nStats_tr_ps;
nSTATS_all_tr{4,1} = nStats_tr_mlp;
nSTATS_all_tr{5,1} = nStats_tr_elm;
nSTATS_all_tr{6,1} = nStats_tr_svc;
nSTATS_all_tr{7,1} = nStats_tr_lssvc;
nSTATS_all_tr{8,1} = nStats_tr_mlm;

nSTATS_all_ts{1,1} = nStats_ts_ols;
nSTATS_all_ts{2,1} = nStats_ts_bay;
nSTATS_all_ts{3,1} = nStats_ts_ps;
nSTATS_all_ts{4,1} = nStats_ts_mlp;
nSTATS_all_ts{5,1} = nStats_ts_elm;
nSTATS_all_ts{6,1} = nStats_ts_svc;
nSTATS_all_ts{7,1} = nStats_ts_lssvc;
nSTATS_all_ts{8,1} = nStats_ts_mlm;

%% BOXPLOT OF TRAIN / TEST / REJECT OPTION ACCURACIES

class_stats_ncomp(nSTATS_all_tr,NAMES);
class_stats_ncomp(nSTATS_all_ts,NAMES);

%% SAVE DATA

save(OPT.file);

%% END