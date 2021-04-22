%% Machine Learning ToolBox

% Test For Wall-Following DataBase and K2NN classifier
% sequential try-and-error: 1 kernel , v1, sigma
% Author: David Nascimento Coelho
% Last Update: 2019/11/29

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% GENERAL DEFINITIONS

% General options' structure

OPT.prob = 22;              % Which problem will be solved / used
OPT.prob2 = 01;             % More details about a specific data set
OPT.norm = 03;              % Normalization definition
OPT.lbl = 01;               % Labeling definition
OPT.Nr = 02;                % Number of repetitions of the algorithm
OPT.hold = 03;              % Hold out method
OPT.ptrn = 0.73;            % Percentage of samples for training
OPT.file = 'WF_k2nn_3_24';  % file where all the variables will be saved

%% DATA LOADING AND PRE-PROCESSING

DATA = data_class_loading(OPT);     % Load Data Set
% DATA = normalize(DATA,OPT);         % normalize the attributes' matrix
DATA = label_encode(DATA,OPT);      % encode labels for the problem

%% HYPERPARAMETERS - DEFAULT

HP.Ss = 01;                 % Sparsification strategy
HP.Us = 01;                 % Update strategy
HP.Ps = 01;                 % Prunning strategy
HP.Dm = 01;                 % Design Method
HP.v1 = 0.00007;                  % Sparseness
HP.Ktype = 1;          	% Kernel Type
HP.sigma = 100;              % Comprehensiveness of kernel

%% ACCUMULATORS

NAMES = {'train','test'};     	% Acc of names for plots
PAR_acc = cell(OPT.Nr,1);     	% Acc of Parameters and Hyperparameters
STATS_tr_acc = cell(OPT.Nr,1);	% Acc of Statistics of training data
STATS_ts_acc = cell(OPT.Nr,1); 	% Acc of Statistics of test data
nSTATS_all = cell(2,1);      	% Acc of General statistics

accuracy = zeros(OPT.Nr,1);
num_prot = zeros(OPT.Nr,1);

%% HOLD OUT (Sequential Data)

DATAtr.input = DATA.input(:,1:4000);
DATAtr.output = DATA.output(:,1:4000);
DATAtr.lbl = DATA.lbl(:,1:4000);

DATAts.input = DATA.input(:,4001:end);
DATAts.output = DATA.output(:,4001:end);
DATAts.lbl = DATA.lbl(:,4001:end);

%% TRAINING / TEST / STATISTICS

disp('Begin Algorithm');

for r = 1:OPT.Nr,

% %%%%%%%%%%%%%% CLASSIFIER'S TRAINING %%%%%%%%%%%%%%%%%%%

PAR_acc{r} = k2nn_train(DATAtr,HP);	% Calculate and acc parameters

% %%%%%%%%% CLASSIFIER'S TEST AND STATISTICS %%%%%%%%%%%%%

[OUTtr] = k2nn_classify(DATAtr,PAR_acc{r});      	% Outputs with training data
STATS_tr_acc{r} = class_stats_1turn(DATAtr,OUTtr);	% Results with training data

[OUTts] = k2nn_classify(DATAts,PAR_acc{r});     	% Outputs with test data
STATS_ts_acc{r} = class_stats_1turn(DATAts,OUTts);	% Results with test data

% %%%%%%%%% CURVE Accuracy x 1 - prototypes %%%%%%%%%%%%%

accuracy(r) = STATS_ts_acc{r}.acc;
[~,num_prot(r)] = size(PAR_acc{r}.Cx);

end

disp('Finish Algorithm')
display(datestr(now));

%% RESULTS / STATISTICS

% Statistics for n turns

nSTATS_tr = class_stats_nturns(STATS_tr_acc);
nSTATS_ts = class_stats_nturns(STATS_ts_acc);

% Get all Statistics in one Cell

nSTATS_all{1,1} = nSTATS_tr;
nSTATS_all{2,1} = nSTATS_ts;

%% BOXPLOT OF TRAINING AND TEST ACCURACIES

class_stats_ncomp(nSTATS_all,NAMES);

%% PLOT: Accuracy x 1 - % of prototypes

Ntr = 4000;
perc_prot = (num_prot/Ntr);

figure;
plot((1-perc_prot),accuracy,'k.');
title('Number of prototypes x accuracy')
xlabel('1 - % of prototypes');
ylabel('accuracy');

%% SAVE DATA

% save(OPT.file);

%% END