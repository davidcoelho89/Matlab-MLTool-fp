%% Machine Learning ToolBox

% Test For Wall-Following DataBase and K2NN classifier
% cross validation: try-and-error: 1 kernel, v1, sigma
% Author: David Nascimento Coelho
% Last Update: 2019/12/10

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
OPT.Nr = 05;                % Number of repetitions of the algorithm
OPT.hold = 03;              % Hold out method
OPT.ptrn = 0.73;            % Percentage of samples for training
OPT.file = 'WF_k2nn_2';     % file where all the variables will be saved

%% DATA LOADING AND PRE-PROCESSING

DATA = data_class_loading(OPT);     % Load Data Set
DATA = normalize(DATA,OPT);         % normalize the attributes' matrix
DATA = label_encode(DATA,OPT);      % adjust labels for the problem

%% CHOOSE HYPERPARAMETERS

HP_acc = cell(OPT.Nr,1);    % Init of Acc Hyperparameters of K2NN-1
HP.Ss = 01;                 % Sparsification strategy
HP.Us = 01;                 % Update strategy
HP.Ps = 01;                 % Prunning strategy
HP.Dm = 02;                 % Design Method
HP.v1 = 0.05;            	% Sparseness
HP.Ktype = 02;              % Kernel Type
HP.sigma = 02;              % Comprehensiveness of kernel

%% ACCUMULATORS

NAMES = {'train','test'};     	% Acc of names for plots
DATA_acc = cell(OPT.Nr,1);    	% Acc of Data
PAR_acc = cell(OPT.Nr,1);     	% Acc of Parameters and Hyperparameters
STATS_tr_acc = cell(OPT.Nr,1);	% Acc of Statistics of training data
STATS_ts_acc = cell(OPT.Nr,1);	% Acc of Statistics of test data
nSTATS_all = cell(2,1);      	% Acc of General statistics

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

PAR_acc{r} = k2nn_train(DATAtr,HP);

% %%%%%%%%% CLASSIFIER'S TEST AND STATISTICS %%%%%%%%%%%%%

OUTtr = k2nn_classify(DATAtr,PAR_acc{r});
STATS_tr_acc{r} = class_stats_1turn(DATAtr,OUTtr);

OUTts = k2nn_classify(DATAts,PAR_acc{r});
STATS_ts_acc{r} = class_stats_1turn(DATAts,OUTts);

end

display('Finish Algorithm')
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

%% END