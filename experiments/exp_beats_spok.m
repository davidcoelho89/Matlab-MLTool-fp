%% BEATS DATASET

% Classification Algorithms - SPOK
% Author: David Nascimento Coelho
% Last Update: 2022/02/03

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% CHOOSE EXPERIMENT PARAMETERS

% General options' structure

OPT.Nr = 05;           	% Number of realizations
OPT.alg = 'ols';        % Which classifier will be used
OPT.prob = 40;        	% Beats Data Set
OPT.norm = 2;         	% Normalization definition (balanced training)
OPT.lbl = 1;           	% Labeling definition [-1 +1]
OPT.hold = 2;         	% Hold out method
OPT.ptrn = 0.7;        	% Percentage of samples for training
OPT.file = 'fileX.mat';	% file where all the variables will be saved

% Grid Search Parameters

GSp.fold = 5;           % number of data partitions for cross validation
GSp.type = 1;           % Takes into account just accuracy
GSp.lambda = 0.5;       % Jpbc = Ds + lambda * Err (prototype-based models)

%% CHOOSE FIXED HYPERPARAMETERS 

% SPOK
HP.Ne = 2;          % Maximum number of epochs
HP.Dm = 2;          % Design Method
HP.Ss = 1;          % Sparsification strategy
HP.v1 = 0.1;        % Sparseness parameter 1 
HP.v2 = 0.9;        % Sparseness parameter 2
HP.Us = 1;          % Update strategy
HP.eta = 0.1;       % Update Rate
HP.Ps = 2;          % Prunning strategy
HP.min_score = -10; % Score that leads to prune prototype
HP.max_prot = Inf;  % Max number of prototypes
HP.min_prot = 1;    % Min number of prototypes
HP.Von = 0;         % Enable / disable video
HP.K = 1;           % Number of nearest neighbors (classify)
HP.knn_type = 1;    % Majority voting for knn
HP.Ktype = 2;       % Kernel Type (gaussian)
HP.sig2n = 0.001;   % Kernel regularization parameter
HP.sigma = 2;       % Kernel width (gaussian)
HP.alpha = 1;       % Dot product multiplier
HP.theta = 1;       % Dot product add cte 
HP.gamma = 2;       % Polynomial order

%% CHOOSE HYPERPARAMETERS TO BE OPTIMIZED

HPgs = HP;

% Can put here vectors of hyperparameters 
% to be optimized. Ex: HPgs.eta = 0.01:0.01:0.1

%% ACCUMULATORS

NAMES = {'train','test'};           % Acc of names for plots
DATA_acc = cell(OPT.Nr,1);       	% Acc of Data
PAR_acc = cell(OPT.Nr,1);         	% Acc of Parameters and Hyperparameters
STATS_tr_acc = cell(OPT.Nr,1);   	% Acc of Statistics of training data
STATS_ts_acc = cell(OPT.Nr,1);   	% Acc of Statistics of test data
nSTATS_all = cell(2,1);             % Acc of General statistics

%% HANDLERS FOR CLASSIFICATION FUNCTIONS

class_name = upper(OPT.alg);

str_train = strcat(lower(OPT.alg),'_train');
class_train = str2func(str_train);

str_test = strcat(lower(OPT.alg),'_classify');
class_test = str2func(str_test);

%% DATA LOADING, PRE-PROCESSING, VISUALIZATION

DATA = data_class_loading(OPT);     % Load Data Set

DATA = label_encode(DATA,OPT);      % adjust labels for the problem

plot_data_pairplot(DATA);         % See pairplot of attributes

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

% Get Normalization Parameters

PARnorm = normalize_fit(DATAtr,OPT);

% Training data normalization

DATAtr = normalize_transform(DATAtr,PARnorm);

% Test data normalization

DATAts = normalize_transform(DATAts,PARnorm);

% Adjust Values for video function

DATA = normalize_transform(DATA,PARnorm);
DATAtr.Xmax = max(DATA.input,[],2);  % max value
DATAtr.Xmin = min(DATA.input,[],2);  % min value
DATAtr.Xmed = mean(DATA.input,2);    % mean value
DATAtr.Xdp = std(DATA.input,[],2);   % std value

% %%%%%%%%%%%%%% SHUFFLE TRAINING DATA %%%%%%%%%%%%%%%%%%%

I = randperm(size(DATAtr.input,2));
DATAtr.input = DATAtr.input(:,I);
DATAtr.output = DATAtr.output(:,I);
DATAtr.lbl = DATAtr.lbl(:,I);

% %%%%%%%%%%% HYPERPARAMETER OPTIMIZATION %%%%%%%%%%%%%%%%

% Using Grid Search and Cross-Validation
% HP = grid_search_cv(DATAtr,HPgs,class_train,class_test,GSp);

% %%%%%%%%%%%%%% CLASSIFIER'S TRAINING %%%%%%%%%%%%%%%%%%%

% Calculate model's parameters
PAR_acc{r} = class_train(DATAtr,HP);

% %%%%%%%%% CLASSIFIER'S TEST AND STATISTICS %%%%%%%%%%%%%

% Results and Statistics with training data
OUTtr = class_test(DATAtr,PAR_acc{r});
STATS_tr_acc{r} = class_stats_1turn(DATAtr,OUTtr);

% Results and Statistics with test data
OUTts = class_test(DATAts,PAR_acc{r});
STATS_ts_acc{r} = class_stats_1turn(DATAts,OUTts);

end

disp('Finish Algorithm')
disp(datestr(now));

%% RESULTS / STATISTICS

% Statistics for n turns

nSTATS_tr = class_stats_nturns(STATS_tr_acc);
nSTATS_ts = class_stats_nturns(STATS_ts_acc);

% Get all Statistics in one Cell

nSTATS_all{1,1} = nSTATS_tr;
nSTATS_all{2,1} = nSTATS_ts;

% Compare Training and Test Statistics

class_stats_ncomp(nSTATS_all,NAMES); 

%% END




























