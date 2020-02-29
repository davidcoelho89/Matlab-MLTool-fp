%% Machine Learning ToolBox

% Clustering Algorithms - General Tests
% Author: David Nascimento Coelho
% Last Update: 2019/09/25

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% GENERAL DEFINITIONS

% General options' structure

OPT.prob = 06;              % Which problem will be solved / used
OPT.prob2 = 01;             % More details about a specific data set
OPT.norm = 3;               % Normalization definition
OPT.lbl = 0;                % Labeling definition
OPT.Nr = 05;                % Number of repetitions of each algorithm
OPT.hold = 2;               % Hold out method
OPT.ptrn = 0.8;             % Percentage of samples for training
OPT.file = 'fileX.mat';     % file where all the variables will be saved

%% DATA LOADING AND PRE-PROCESSING

DATA = data_class_loading(OPT);     % Load Data Set

Nc = length(unique(DATA.output));	% get number of classes

[p,N] = size(DATA.input);           % get number of attributes and samples

DATA = normalize(DATA,OPT);         % normalize the attributes' matrix

DATA = label_adjust(DATA,OPT);      % adjust labels for the problem

%% ACCUMULATORS

% Names Accumulator

NAMES = {'km','k2nn','ksomef','ksomgd','ksomps', ...
         'lvq','ng','som','wta'};

% General statistics cell

nSTATS_all = cell(10,1);

% Statistics and Parameters Accumulators

STATS_km = cell(OPT.Nr,1);      % Acc of Statistics
PAR_km = cell(OPT.Nr,1);        % Acc of Parameters and Hyperparameters
STATS_k2nn = cell(OPT.Nr,1); 	% Acc of Statistics
PAR_k2nn = cell(OPT.Nr,1);   	% Acc of Parameters and Hyperparameters
STATS_ksomef = cell(OPT.Nr,1);	% Acc of Statistics
PAR_ksomef = cell(OPT.Nr,1);   	% Acc of Parameters and Hyperparameters
STATS_ksomgd = cell(OPT.Nr,1);	% Acc of Statistics
PAR_ksomgd = cell(OPT.Nr,1);   	% Acc of Parameters and Hyperparameters
STATS_ksomps = cell(OPT.Nr,1); 	% Acc of Statistics
PAR_ksomps = cell(OPT.Nr,1);   	% Acc of Parameters and Hyperparameters
STATS_lvq = cell(OPT.Nr,1);     % Acc of Statistics
PAR_lvq = cell(OPT.Nr,1);   	% Acc of Parameters and Hyperparameters
STATS_ng = cell(OPT.Nr,1);      % Acc of Statistics
PAR_ng = cell(OPT.Nr,1);        % Acc of Parameters and Hyperparameters
STATS_som = cell(OPT.Nr,1);     % Acc of Statistics
PAR_som = cell(OPT.Nr,1);   	% Acc of Parameters and Hyperparameters
STATS_wta = cell(OPT.Nr,1);     % Acc of Statistics
PAR_wta = cell(OPT.Nr,1);   	% Acc of Parameters and Hyperparameters

%% CLUSTERING

display('Begin Algorithms');

for r = 1:OPT.Nr;

% %%%%%%%%% DISPLAY REPETITION AND DURATION %%%%%%%%%%%%%%

display(r);
display(datestr(now));

% %%%%%%%%%%%% CLUSTERING AND LABELING %%%%%%%%%%%%%%%%%%%

OUT_KM = kmeans_cluster(DATA);
PAR_km{r} = kmeans_label(DATA,OUT_KM);

PAR_k2nn{r} = k2nn_train(DATA);

OUT_KSOMEF = ksom_ef_cluster(DATA);
PAR_ksomef{r} = ksom_ef_label(DATA,OUT_KSOMEF);

OUT_KSOMGD = ksom_gd_cluster(DATA);
PAR_ksomgd{r} = ksom_gd_label(DATA,OUT_KSOMGD);

OUT_KSOMPS = ksom_ps_cluster(DATA);
PAR_ksomps{r} = ksom_ps_label(DATA,OUT_KSOMPS);

OUT_LVQ = lvq_cluster(DATA);
PAR_lvq{r} = lvq_label(DATA,OUT_LVQ);

OUT_NG = ng_cluster(DATA);
PAR_ng{r} = ng_label(DATA,OUT_NG);

OUT_SOM = som_cluster(DATA);
PAR_som{r} = som_label(DATA,OUT_SOM);

OUT_WTA = wta_cluster(DATA);
PAR_wta{r} = wta_label(DATA,OUT_WTA);

% %%%%%%%%%%%%%%%%%%% STATISTICS %%%%%%%%%%%%%%%%%%%%%%%%%

STATS_km{r} = cluster_stats_1turn(DATA,PAR_km{r});
STATS_k2nn{r} = cluster_stats_1turn(DATA,PAR_k2nn{r});
STATS_ksomef{r} = cluster_stats_1turn(DATA,PAR_ksomef{r});
STATS_ksomgd{r} = cluster_stats_1turn(DATA,PAR_ksomgd{r});
STATS_ksomps{r} = cluster_stats_1turn(DATA,PAR_ksomps{r});
STATS_lvq{r} = cluster_stats_1turn(DATA,PAR_lvq{r});
STATS_ng{r} = cluster_stats_1turn(DATA,PAR_ng{r});
STATS_som{r} = cluster_stats_1turn(DATA,PAR_som{r});
STATS_wta{r} = cluster_stats_1turn(DATA,PAR_wta{r});

end

display('Finish Algorithms')
display(datestr(now));

%% RESULTS / STATISTICS

% Statistics for n turns

nSTATS_km = cluster_stats_nturns(STATS_km);
nSTATS_k2nn = cluster_stats_nturns(STATS_k2nn);
nSTATS_ksomef = cluster_stats_nturns(STATS_ksomef);
nSTATS_ksomgd = cluster_stats_nturns(STATS_ksomgd);
nSTATS_ksomps = cluster_stats_nturns(STATS_ksomps);
nSTATS_lvq = cluster_stats_nturns(STATS_lvq);
nSTATS_ng = cluster_stats_nturns(STATS_ng);
nSTATS_som = cluster_stats_nturns(STATS_som);
nSTATS_wta = cluster_stats_nturns(STATS_wta);

% Get all Statistics in one Cell

nSTATS_all{1,1} = nSTATS_km;
nSTATS_all{2,1} = nSTATS_k2nn;
nSTATS_all{3,1} = nSTATS_ksomef;
nSTATS_all{4,1} = nSTATS_ksomgd;
nSTATS_all{5,1} = nSTATS_ksomps;
nSTATS_all{6,1} = nSTATS_lvq;
nSTATS_all{7,1} = nSTATS_ng;
nSTATS_all{8,1} = nSTATS_som;
nSTATS_all{9,1} = nSTATS_wta;

%% GRAPHICS



%% END