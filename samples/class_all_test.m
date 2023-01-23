%% Machine Learning ToolBox

% Classification Algorithms - General Tests
% Author: David Nascimento Coelho
% Last Update: 2020/02/13

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% GENERAL DEFINITIONS

% General options' structure

OPT.prob = 06;              % Which problem will be solved / used
OPT.prob2 = 01;             % More details about a specific data set
OPT.norm = 3;               % Normalization definition
OPT.lbl = 1;                % Labeling definition
OPT.Nr = 05;                % Number of repetitions of each algorithm
OPT.hold = 2;               % Hold out method
OPT.ptrn = 0.8;             % Percentage of samples for training
OPT.file = 'fileX.mat';     % file where all the variables will be saved

%% DATA LOADING AND PRE-PROCESSING

DATA = data_class_loading(OPT);     % Load Data Set

Nc = length(unique(DATA.output));   % get number of classes

[p,N] = size(DATA.input);           % get number of attributes and samples

DATA = normalize(DATA,OPT);         % normalize the attributes' matrix

DATA = label_encode(DATA,OPT);      % adjust labels for the problem

%% ACCUMULATORS

% Names Accumulator

NAMES = {'adal','elm','gauss','k2nn','km','knn','kqd','krr','ksomef',...
         'ksomgd','ksomps','lms','lssvc','lvq','mlm','mlp','ng','ols',...
         'ps', 'rbf','rls','som','svc','wta'};

% General statistics cells

nSTATS_all = cell(length(NAMES),1);     % Comparison of stats for n turns
STATS_1_all = cell(length(NAMES),1);	% Comparison of stats for 1 turns
STATS_1_comp = cell(OPT.Nr,1);          % Hold comparison for each turn

% Statistics Accumulators

STATS_adal = cell(OPT.Nr,1);      	% Acc of Statistics of Adaline
STATS_elm = cell(OPT.Nr,1);       	% Acc of Statistics of ELM
STATS_gauss = cell(OPT.Nr,1);      	% Acc of Statistics of Gaussian
STATS_k2nn = cell(OPT.Nr,1);       	% Acc of Statistics of K2nn
STATS_km = cell(OPT.Nr,1);       	% Acc of Statistics of K-means
STATS_knn = cell(OPT.Nr,1);       	% Acc of Statistics of KNN
STATS_kqd = cell(OPT.Nr,1);       	% Acc of Statistics of KQD
STATS_krr = cell(OPT.Nr,1);       	% Acc of Statistics of KRR
STATS_ksom_ef = cell(OPT.Nr,1);   	% Acc of Statistics of KSOM-EF
STATS_ksom_gd = cell(OPT.Nr,1);  	% Acc of Statistics of KSOM-GD
STATS_ksom_ps = cell(OPT.Nr,1);  	% Acc of Statistics of KSOM-PS
STATS_lms = cell(OPT.Nr,1);       	% Acc of Statistics of LMS
STATS_lssvc = cell(OPT.Nr,1);     	% Acc of Statistics of LSSVC
STATS_lvq = cell(OPT.Nr,1);       	% Acc of Statistics of LVQ
STATS_mlm = cell(OPT.Nr,1);       	% Acc of Statistics of MLM
STATS_mlp = cell(OPT.Nr,1);       	% Acc of Statistics of MLP
STATS_ng = cell(OPT.Nr,1);       	% Acc of Statistics of NG
STATS_ols = cell(OPT.Nr,1);       	% Acc of Statistics of OLS
STATS_ps = cell(OPT.Nr,1);       	% Acc of Statistics of PS
STATS_rbf = cell(OPT.Nr,1);       	% Acc of Statistics of RBF
STATS_rls = cell(OPT.Nr,1);       	% Acc of Statistics of RLS
STATS_som = cell(OPT.Nr,1);       	% Acc of Statistics of SOM
STATS_svc = cell(OPT.Nr,1);       	% Acc of Statistics of SVC
STATS_wta = cell(OPT.Nr,1);       	% Acc of Statistics of WTA

%% HOLD OUT / CROSS VALIDATION / TRAINING / TEST

for r = 1:OPT.Nr
    
% %%%%%%%%% DISPLAY REPETITION AND DURATION %%%%%%%%%%%%%%

disp(r);
display(datestr(now));

% %%%%%%%%%%%%%% SHUFFLE AND HOLD OUT %%%%%%%%%%%%%%%%%%%%

% Shuffle data

I = randperm(N);
DATA.input = DATA.input(:,I); 
DATA.output = DATA.output(:,I);
DATA.lbl = DATA.lbl(:,I);

% Hold out

[DATAho] = hold_out(DATA,OPT);
DATAtr = DATAho.DATAtr;
DATAts = DATAho.DATAts;

% %%%%%%%%%%%%%% CLASSIFIERS' TRAINING %%%%%%%%%%%%%%%%%%%

[ADALp] = adaline_train(DATAtr);
[ELMp] = elm_train(DATAtr);
[BAYp] = gauss_train(DATAtr);
[K2NNp] = k2nn_train(DATAtr);
[KMp] = kmeans_train(DATAtr);
[KNNp] = knn_train(DATAtr);
[KQDp] = kqd_train(DATAtr);
[KRRp] = krr_train(DATAtr);
[KSOMEFp] = ksom_ef_train(DATAtr);
[KSOMGDp] = ksom_gd_train(DATAtr);
[KSOMPSp] = ksom_ps_train(DATAtr);
[LMSp] = lms_train(DATAtr);
[LSSVCp] = lssvc_train(DATAtr);
[LVQp] = lvq_train(DATAtr);
[MLMp] = mlm_train(DATAtr);
[MLPp] = mlp_train(DATAtr);
[NGp] = ng_train(DATAtr);
[OLSp] = ols_train(DATAtr);
[PSp] = ps_train(DATAtr);
[RBFp] = rbf_train(DATAtr);
[RLSp] = rls_train(DATAtr);
[SOMp] = som_train(DATAtr);
[SVCp] = svc_train(DATAtr);
[WTAp] = wta_train(DATAtr);

% %%%%%%%%%%%%%%%%% CLASSIFIERS' TEST %%%%%%%%%%%%%%%%%%%%

[OUT_adal] = adaline_classify(DATAts,ADALp);
[OUT_elm] = elm_classify(DATAts,ELMp);
[OUT_bay] = gauss_classify(DATAts,BAYp);
[OUT_k2nn] = k2nn_classify(DATAts,K2NNp);
[OUT_kmeans] = kmeans_classify(DATAts,KMp);
[OUT_knn] = knn_classify(DATAts,KNNp);
[OUT_kqd] = kqd_classify(DATAts,KQDp);
[OUT_krr] = krr_classify(DATAts,KRRp);
[OUT_ksom_ef] = ksom_ef_classify(DATAts,KSOMEFp);
[OUT_ksom_gd] = ksom_gd_classify(DATAts,KSOMGDp);
[OUT_ksom_ps] = ksom_ps_classify(DATAts,KSOMPSp);
[OUT_lms] = lms_classify(DATAts,LMSp);
[OUT_lssvc] = lssvc_classify(DATAts,LSSVCp);
[OUT_lvq] = lvq_classify(DATAts,LVQp);
[OUT_mlm] = mlm_classify(DATAts,MLMp);
[OUT_mlp] = mlp_classify(DATAts,MLPp);
[OUT_ng] = ng_classify(DATAts,NGp);
[OUT_ols] = ols_classify(DATAts,OLSp);
[OUT_ps] = ps_classify(DATAts,PSp);
[OUT_rbf] = rbf_classify(DATAts,RBFp);
[OUT_rls] = rls_classify(DATAts,RLSp);
[OUT_som] = som_classify(DATAts,SOMp);
[OUT_svc] = svc_classify(DATAts,SVCp);
[OUT_wta] = wta_classify(DATAts,WTAp);

% %%%%%%%%%%%%%% CLASSIFIERS' STATISTICS %%%%%%%%%%%%%%%%%

STATS_adal{r} = class_stats_1turn(DATAts,OUT_adal);
STATS_elm{r} = class_stats_1turn(DATAts,OUT_elm);
STATS_gauss{r} = class_stats_1turn(DATAts,OUT_bay);
STATS_k2nn{r} = class_stats_1turn(DATAts,OUT_k2nn);
STATS_km{r} = class_stats_1turn(DATAts,OUT_kmeans);
STATS_knn{r} = class_stats_1turn(DATAts,OUT_knn);
STATS_kqd{r} = class_stats_1turn(DATAts,OUT_kqd);
STATS_krr{r} = class_stats_1turn(DATAts,OUT_krr);
STATS_ksom_ef{r} = class_stats_1turn(DATAts,OUT_ksom_ef);
STATS_ksom_gd{r} = class_stats_1turn(DATAts,OUT_ksom_gd);
STATS_ksom_ps{r} = class_stats_1turn(DATAts,OUT_ksom_ps);
STATS_lms{r} = class_stats_1turn(DATAts,OUT_lms);
STATS_lssvc{r} = class_stats_1turn(DATAts,OUT_lssvc);
STATS_lvq{r} = class_stats_1turn(DATAts,OUT_lvq);
STATS_mlm{r} = class_stats_1turn(DATAts,OUT_mlm);
STATS_mlp{r} = class_stats_1turn(DATAts,OUT_mlp);
STATS_ng{r} = class_stats_1turn(DATAts,OUT_ng);
STATS_ols{r} = class_stats_1turn(DATAts,OUT_ols);
STATS_ps{r} = class_stats_1turn(DATAts,OUT_ps);
STATS_rbf{r} = class_stats_1turn(DATAts,OUT_rbf);
STATS_rls{r} = class_stats_1turn(DATAts,OUT_rls);
STATS_som{r} = class_stats_1turn(DATAts,OUT_som);
STATS_svc{r} = class_stats_1turn(DATAts,OUT_svc);
STATS_wta{r} = class_stats_1turn(DATAts,OUT_wta);

% %%%%%%%%%%%%%% CLASSIFIERS' COMPARISON %%%%%%%%%%%%%%%%%

% Hold Statistics for 1 turn in one cell

STATS_1_all{1,1} = STATS_adal{r};
STATS_1_all{2,1} = STATS_elm{r};
STATS_1_all{3,1} = STATS_gauss{r};
STATS_1_all{4,1} = STATS_k2nn{r};
STATS_1_all{5,1} = STATS_km{r};
STATS_1_all{6,1} = STATS_knn{r};
STATS_1_all{7,1} = STATS_kqd{r};
STATS_1_all{8,1} = STATS_krr{r};
STATS_1_all{9,1} = STATS_ksom_ef{r};
STATS_1_all{10,1} = STATS_ksom_gd{r};
STATS_1_all{11,1} = STATS_ksom_ps{r};
STATS_1_all{12,1} = STATS_lms{r};
STATS_1_all{13,1} = STATS_lssvc{r};
STATS_1_all{14,1} = STATS_lvq{r};
STATS_1_all{15,1} = STATS_mlm{r};
STATS_1_all{16,1} = STATS_mlp{r};
STATS_1_all{17,1} = STATS_ng{r};
STATS_1_all{18,1} = STATS_ols{r};
STATS_1_all{19,1} = STATS_ps{r};
STATS_1_all{20,1} = STATS_rbf{r};
STATS_1_all{21,1} = STATS_rls{r};
STATS_1_all{22,1} = STATS_som{r};
STATS_1_all{23,1} = STATS_svc{r};
STATS_1_all{24,1} = STATS_wta{r};

% Compare classifiers for 1 turn

STATS_1_comp = class_stats_1comp(STATS_1_all,NAMES);

end

%% RESULTS / STATISTICS

% Statistics for n turns

nSTATS_all{1,1} = class_stats_nturns(STATS_adal);
nSTATS_all{2,1} = class_stats_nturns(STATS_elm);
nSTATS_all{3,1} = class_stats_nturns(STATS_gauss);
nSTATS_all{5,1} = class_stats_nturns(STATS_k2nn);
nSTATS_all{4,1} = class_stats_nturns(STATS_km);
nSTATS_all{6,1} = class_stats_nturns(STATS_knn);
nSTATS_all{7,1} = class_stats_nturns(STATS_kqd);
nSTATS_all{8,1} = class_stats_nturns(STATS_krr);
nSTATS_all{9,1} = class_stats_nturns(STATS_ksom_ef);
nSTATS_all{10,1} = class_stats_nturns(STATS_ksom_gd);
nSTATS_all{11,1} = class_stats_nturns(STATS_ksom_ps);
nSTATS_all{12,1} = class_stats_nturns(STATS_lms);
nSTATS_all{13,1} = class_stats_nturns(STATS_lssvc);
nSTATS_all{14,1} = class_stats_nturns(STATS_lvq);
nSTATS_all{15,1} = class_stats_nturns(STATS_mlm);
nSTATS_all{16,1} = class_stats_nturns(STATS_mlp);
nSTATS_all{17,1} = class_stats_nturns(STATS_ng);
nSTATS_all{18,1} = class_stats_nturns(STATS_ols);
nSTATS_all{19,1} = class_stats_nturns(STATS_ps);
nSTATS_all{20,1} = class_stats_nturns(STATS_rbf);
nSTATS_all{21,1} = class_stats_nturns(STATS_rls);
nSTATS_all{22,1} = class_stats_nturns(STATS_som);
nSTATS_all{23,1} = class_stats_nturns(STATS_svc);
nSTATS_all{24,1} = class_stats_nturns(STATS_wta);

% Compare Classifiers

class_stats_ncomp(nSTATS_all,NAMES);

%% SAVE DATA

% save(OPT.file);

%% END