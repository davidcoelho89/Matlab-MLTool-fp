%% Machine Learning ToolBox

% Feature Extraction Algorithms
% Author: David Nascimento Coelho
% Last Update: 2020/02/02

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% GENERAL DEFINITIONS

% General options' structure

OPT.prob = 06;              % Which problem will be solved / used
OPT.norm = 3;               % Normalization definition
OPT.lbl = 1;                % Labeling definition
OPT.Nr = 10;              	% Number of repetitions of the algorithm
OPT.hold = 2;               % Hold out method
OPT.ptrn = 0.7;             % Percentage of samples for training

%% DATA LOADING AND PRE-PROCESSING

DATA = data_class_loading(OPT);     % Load Data Set
DATA = normalize(DATA,OPT);         % normalize the attributes' matrix
DATA = label_encode(DATA,OPT);      % adjust labels for the problem

%% HIPERPARAMETERS

OLShp.aprox = 1;        % Type of aproximation (same for all classifiers)

PCAhp.tol = 0.95;       % tolerance of explained value
PCAhp.rem = 1;          % remove mean of data

LDAhp.tol = 0.999;      % tolerance of explained value
LDAhp.rem = 1;          % remove mean of data

KPCAhp.tol = 1;         % tolerance of explained value
KPCAhp.rem = 1;         % remove mean of data
KPCAhp.Ktype = 2;       % kernel type
KPCAhp.sig2n = 0.001;   % kernel regularization parameter
KPCAhp.sigma = 2;       % kernel hyperparameter

KLDAhp.tol = 1;         % tolerance of explained value
KLDAhp.rem = 1;         % remove mean of data
KLDAhp.Ktype = 2;       % kernel type
KLDAhp.sig2n = 0.001;   % kernel regularization parameter
KLDAhp.sigma = 2;       % kernel hyperparameter

%% ACCUMULATORS

% Acc of names for plots

NAMES = {'Original','PCA','LDA','KPCA','KLDA'};

% Acc of labels and data division

DATA_acc = cell(OPT.Nr,1);       	% Acc of Data

% General statistics cell

nSTATS_all_tr = cell(length(NAMES),1);
nSTATS_all_ts = cell(length(NAMES),1);

% Feature Selection

PCAp_acc = cell(OPT.Nr,1);          % Acc Parameters of PCA

LDAp_acc = cell(OPT.Nr,1);          % Acc Parameters of LDA

KPCAp_acc = cell(OPT.Nr,1);         % Acc Parameters of KPCA

KLDAp_acc = cell(OPT.Nr,1);         % Acc Parameters of KLDA

% Classifiers

OLSp1_acc = cell(OPT.Nr,1);      	% Acc Parameters of OLS (no feat select)
ols1_out_tr = cell(OPT.Nr,1);   	% Acc of training data output
ols1_stats_tr = cell(OPT.Nr,1);   	% Acc of statistics from training
ols1_out_ts = cell(OPT.Nr,1);     	% Acc of test data output
ols1_stats_ts = cell(OPT.Nr,1);   	% Acc of statistics from test

OLSp2_acc = cell(OPT.Nr,1);      	% Acc Parameters of OLS 2 (pca)
ols2_out_tr = cell(OPT.Nr,1);     	% Acc of training data output
ols2_stats_tr = cell(OPT.Nr,1);  	% Acc of statistics from training
ols2_out_ts = cell(OPT.Nr,1);      	% Acc of test data output
ols2_stats_ts = cell(OPT.Nr,1);   	% Acc of statistics from test

OLSp3_acc = cell(OPT.Nr,1);      	% Acc Parameters of OLS 3 (lda)
ols3_out_tr = cell(OPT.Nr,1);     	% Acc of training data output
ols3_stats_tr = cell(OPT.Nr,1);  	% Acc of statistics from training
ols3_out_ts = cell(OPT.Nr,1);      	% Acc of test data output
ols3_stats_ts = cell(OPT.Nr,1);   	% Acc of statistics from test

OLSp4_acc = cell(OPT.Nr,1);      	% Acc Parameters of OLS 4 (kpca)
ols4_out_tr = cell(OPT.Nr,1);     	% Acc of training data output
ols4_stats_tr = cell(OPT.Nr,1);  	% Acc of statistics from training
ols4_out_ts = cell(OPT.Nr,1);      	% Acc of test data output
ols4_stats_ts = cell(OPT.Nr,1);   	% Acc of statistics from test

OLSp5_acc = cell(OPT.Nr,1);      	% Acc Parameters of OLS 5 (klda)
ols5_out_tr = cell(OPT.Nr,1);     	% Acc of training data output
ols5_stats_tr = cell(OPT.Nr,1);  	% Acc of statistics from training
ols5_out_ts = cell(OPT.Nr,1);      	% Acc of test data output
ols5_stats_ts = cell(OPT.Nr,1);   	% Acc of statistics from test

%% HOLD OUT / TRAINING / TEST / STATISTICS

for r = 1:OPT.Nr

% Repetition

display(r);
display(datestr(now));
    
% Hold out

DATA_acc{r} = hold_out(DATA,OPT);   % Hold Out Function
DATAtr = DATA_acc{r}.DATAtr;        % Training Data
DATAts = DATA_acc{r}.DATAts;      	% Test Data

% Shuffle

I = randperm(size(DATAtr.input,2));
DATAtr.input = DATAtr.input(:,I);
DATAtr.output = DATAtr.output(:,I);
DATAtr.lbl = DATAtr.lbl(:,I);

% Generate matrix and apply transformation for data set

PCAp_acc{r} = pca_feature(DATAtr,PCAhp);
DATAtr_PCA.input = PCAp_acc{r}.input;
DATAtr_PCA.output = PCAp_acc{r}.output;

LDAp_acc{r} = lda_feature(DATAtr,LDAhp);
DATAtr_LDA.input = LDAp_acc{r}.input;
DATAtr_LDA.output = LDAp_acc{r}.output;

KPCAp_acc{r} = kpca_feature(DATAtr,KPCAhp);
DATAtr_KPCA.input = KPCAp_acc{r}.input;
DATAtr_KPCA.output = KPCAp_acc{r}.output;

KLDAp_acc{r} = klda_feature(DATAtr,KLDAhp);
DATAtr_KLDA.input = KLDAp_acc{r}.input;
DATAtr_KLDA.output = KLDAp_acc{r}.output;

% Train OLS With Modified data

OLSp1_acc{r} = ols_train(DATAtr,OLShp);

OLSp2_acc{r} = ols_train(DATAtr_PCA,OLShp);

OLSp3_acc{r} = ols_train(DATAtr_LDA,OLShp);

OLSp4_acc{r} = ols_train(DATAtr_KPCA,OLShp);

OLSp5_acc{r} = ols_train(DATAtr_KLDA,OLShp);

% Modify test data

DATAts_PCA = pca_transform(DATAts,PCAp_acc{r});

DATAts_LDA = lda_transform(DATAts,LDAp_acc{r});

DATAts_KPCA = kpca_transform(DATAts,KPCAp_acc{r});

DATAts_KLDA = klda_transform(DATAts,KLDAp_acc{r});

% Test Classifiers

ols1_out_tr{r} = ols_classify(DATAtr,OLSp1_acc{r});
ols1_stats_tr{r} = class_stats_1turn(DATAtr,ols1_out_tr{r});
ols1_out_ts{r} = ols_classify(DATAts,OLSp1_acc{r});
ols1_stats_ts{r} = class_stats_1turn(DATAts,ols1_out_ts{r});

ols2_out_tr{r} = ols_classify(DATAtr_PCA,OLSp2_acc{r});
ols2_stats_tr{r} = class_stats_1turn(DATAtr_PCA,ols2_out_tr{r});
ols2_out_ts{r} = ols_classify(DATAts_PCA,OLSp2_acc{r});
ols2_stats_ts{r} = class_stats_1turn(DATAts_PCA,ols2_out_ts{r});

ols3_out_tr{r} = ols_classify(DATAtr_LDA,OLSp3_acc{r});
ols3_stats_tr{r} = class_stats_1turn(DATAtr_LDA,ols3_out_tr{r});
ols3_out_ts{r} = ols_classify(DATAts_LDA,OLSp3_acc{r});
ols3_stats_ts{r} = class_stats_1turn(DATAts_LDA,ols3_out_ts{r});

ols4_out_tr{r} = ols_classify(DATAtr_KPCA,OLSp4_acc{r});
ols4_stats_tr{r} = class_stats_1turn(DATAtr_KPCA,ols4_out_tr{r});
ols4_out_ts{r} = ols_classify(DATAts_KPCA,OLSp4_acc{r});
ols4_stats_ts{r} = class_stats_1turn(DATAts_KPCA,ols4_out_ts{r});

ols5_out_tr{r} = ols_classify(DATAtr_KLDA,OLSp5_acc{r});
ols5_stats_tr{r} = class_stats_1turn(DATAtr_KLDA,ols5_out_tr{r});
ols5_out_ts{r} = ols_classify(DATAts_KLDA,OLSp5_acc{r});
ols5_stats_ts{r} = class_stats_1turn(DATAts_KLDA,ols5_out_ts{r});

end

%% RESULTS / STATISTICS

% Statistics for n turns

nStats1_tr_ols = class_stats_nturns(ols1_stats_tr);
nStats1_ts_ols = class_stats_nturns(ols1_stats_ts);

nStats2_tr_ols = class_stats_nturns(ols2_stats_tr);
nStats2_ts_ols = class_stats_nturns(ols2_stats_ts);

nStats3_tr_ols = class_stats_nturns(ols3_stats_tr);
nStats3_ts_ols = class_stats_nturns(ols3_stats_ts);

nStats4_tr_ols = class_stats_nturns(ols4_stats_tr);
nStats4_ts_ols = class_stats_nturns(ols4_stats_ts);

nStats5_tr_ols = class_stats_nturns(ols5_stats_tr);
nStats5_ts_ols = class_stats_nturns(ols5_stats_ts);

% Get all Statistics in one Cell

nSTATS_all_tr{1,1} = nStats1_tr_ols;
nSTATS_all_tr{2,1} = nStats2_tr_ols;
nSTATS_all_tr{3,1} = nStats3_tr_ols;
nSTATS_all_tr{4,1} = nStats4_tr_ols;
nSTATS_all_tr{5,1} = nStats5_tr_ols;

nSTATS_all_ts{1,1} = nStats1_ts_ols;
nSTATS_all_ts{2,1} = nStats2_ts_ols;
nSTATS_all_ts{3,1} = nStats3_ts_ols;
nSTATS_all_ts{4,1} = nStats4_ts_ols;
nSTATS_all_ts{5,1} = nStats5_ts_ols;

%% BOXPLOT OF TRAIN / TEST / REJECT OPTION ACCURACIES

class_stats_ncomp(nSTATS_all_tr,NAMES);
class_stats_ncomp(nSTATS_all_ts,NAMES);

%% END