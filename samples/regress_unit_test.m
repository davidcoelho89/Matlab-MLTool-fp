%% Machine Learning ToolBox

% Regression Algorithms - Unit Test
% Author: David Nascimento Coelho
% Last Update: 2019/02/13

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window
format long e;  % Output data style (float)

%% GENERAL DEFINITIONS

% General options' structure

OPT.Nr = 10;           	% Number of realizations
OPT.alg = 'ols';        % Which classifier will be used
OPT.prob = 42;        	% Which problem will be solved / used
OPT.prob2 = 01;       	% More details about a specific data set
OPT.norm = 3;         	% Normalization definition
OPT.hold = 1;         	% Hold out method
OPT.ptrn = 0.7;        	% Percentage of samples for training
OPT.hpo = 'none';       % 'grid' ; 'random' ; 'none' ; 'pso'
OPT.file = 'fileX.mat';	% file where all the variables will be saved

OPT.savefile = 0;               % decides if file will be saved
OPT.savevideo = 0;              % decides if video will be saved
OPT.show_specific_stats = 0;    % roc, class boundary, precision-recall

% Metaparameters

MP.max_it = 9;          % Maximum number of iterations (random search)
MP.fold = 5;            % number of data partitions (cross validation)
MP.cost = 1;            % Which cost function will be used
MP.lambda = 0.5;        % Jpbc = Ds + lambda * Err (prototype-based models)

%% CHOOSE FIXED HYPERPARAMETERS 



%% CHOOSE HYPERPARAMETERS TO BE OPTIMIZED



%% ACCUMULATORS

NAMES = {'train','test'};           % Acc of names for plots
DATA_acc = cell(OPT.Nr,1);       	% Acc of Data
PAR_acc = cell(OPT.Nr,1);         	% Acc of Parameters and Hyperparameters
STATS_tr_acc = cell(OPT.Nr,1);   	% Acc of Statistics of training data
STATS_ts_acc = cell(OPT.Nr,1);   	% Acc of Statistics of test data
nSTATS_all = cell(length(NAMES),1,1); % Acc of General statistics

%% HANDLERS FOR CLASSIFICATION FUNCTIONS



%% DATA LOADING, PRE-PROCESSING, VISUALIZATION

DATA = data_class_loading(OPT);     % Load Data Set

%% HOLD OUT / NORMALIZE / SHUFFLE / HPO / TRAINING / TEST / STATISTICS

disp('Begin Algorithm');

for r = 1:OPT.Nr

% %%%%%%%%% DISPLAY REPETITION AND DURATION %%%%%%%%%%%%%%

disp('Turn and Time');
disp(r);
display(datestr(now));

% %%%%%%%%%%%%%%%%%%%% HOLD OUT %%%%%%%%%%%%%%%%%%%%%%%%%%

DATA_acc{r} = hold_out(DATA,OPT);   % Hold Out Function
DATAtr = DATA_acc{r}.DATAtr;        % Training Data
DATAts = DATA_acc{r}.DATAts;      	% Test Data

% %%%%%%%%%%%%%%%%% NORMALIZE DATA %%%%%%%%%%%%%%%%%%%%%%%


    
end

%% RESULTS / STATISTICS



%% GRAPHICS - OF LAST TURN



%% SAVE VARIABLES AND VIDEO



%% END