%% BEATS DATASET

% Classification Algorithms - OLS / MLP / SVM / MLM / ELM / GAUSS /
% Author: David Nascimento Coelho
% Last Update: 2022/04/11

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;   % Output data style (float)

%% CHOOSE EXPERIMENT PARAMETERS

% General options' structure

OPT.Nr = 10;           	% Number of realizations
OPT.alg = 'elm';        % Which classifier will be used
OPT.prob = 40;        	% Beats Data Set
OPT.prob2 = 'real';     % Get specific dataset
OPT.norm = 0;         	% Normalization definition (balanced training)
OPT.lbl = 1;           	% Labeling definition [-1 +1]
OPT.hold = 1;         	% Hold out method
OPT.ptrn = 0.7;        	% Percentage of samples for training
OPT.file = 'fileX.mat';	% file where all the variables will be saved

% Grid Search Parameters

CVp.max_it = 2;         % Maximum number of iterations (random search)
CVp.fold = 5;           % number of data partitions for cross validation
CVp.cost = 1;           % Which cost function will be used
CVp.lambda = 0.5;       % Jpbc = Ds + lambda * Err (prototype-based models)

%% CHOOSE FIXED HYPERPARAMETERS 

% OLS
% HP.aprox = 1;   % Type of approximation

% MLP
% HP.Nh = [50,50]; % 07;	% Number of hidden neurons
% HP.Ne = 500;          	% maximum number of training epochs
% HP.eta = 0.05;          % Learning step
% HP.mom = 0.75;          % Moment Factor
% HP.Nlin = 2;            % Non-linearity
% HP.Von = 0;             % disable video 

% MLM
% HP.dist = 2;        % Gaussian distance
% HP.Ktype = 0;       % Non-kernelized Algorithm
% HP.K = 20;       	% Number of reference points

% SVC
% HP.lambda = 5;      % Regularization Constant
% HP.epsilon = 0.001; % Minimum value of lagrange multipliers
% HP.Ktype = 1;       % Kernel. (1 = linear / 2 = Gaussian)
% HP.sigma = 2;       % Gaussian Kernel std

% ELM
HP.Nh = 2500;       % No. de neuronios na camada oculta
HP.Nlin = 2;    	% Não linearidade ELM (tg hiperb)

% GAUSSIAN (BAYESIAN)
% HP.type = 5;        % Type of gaussian classifier

%% CHOOSE HYPERPARAMETERS TO BE OPTIMIZED

HPgs = HP;

% Can put here vectors of hyperparameters 
% to be optimized. Ex: HPgs.eta = 0.01:0.01:0.1

% SVC
% HPgs.lambda = [0.5 5 10 25 100 500];
% HPgs.Ktype = 2;
% HPgs.sigma = [0.01 0.05 0.5 5 25 100 500];

% ELM
% HPgs.Nh = [1500,2500]; % Grid Search
% HPgs.Nh = [1500,1800,2000,2200,2500]; % Random Search
% HPgs.Nlin = [1,2]; % sigmoid or hyperbolic tangent

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

if (strcmp(OPT.prob2,'original'))
    DATA = data_class_loading(OPT);     % Load Data Set
    DATA = label_encode(DATA,OPT);      % adjust labels for the problem
    % plot_data_pairplot(DATA);         % See pairplot of attributes
elseif (strcmp(OPT.prob2,'artificial'))
    prob_aux = 'artificial';
elseif (strcmp(OPT.prob2,'real'))
    prob_aux = 'real';
end

%% HOLD OUT / NORMALIZE / SHUFFLE / HPO / TRAINING / TEST / STATISTICS

disp('Begin Algorithm');

for r = 1:OPT.Nr

% %%%%%%%%% DISPLAY REPETITION AND DURATION %%%%%%%%%%%%%%

disp('Turn and Time');
disp(r);
display(datestr(now));

% %%%%%%%%%%%%%%%%%%%% HOLD OUT %%%%%%%%%%%%%%%%%%%%%%%%%%

if(strcmp(OPT.prob2,'original'))
    DATA_acc{r} = hold_out(DATA,OPT);   % Hold Out Function
    DATAtr = DATA_acc{r}.DATAtr;        % Training Data
    DATAts = DATA_acc{r}.DATAts;      	% Test Data

elseif (strcmp(prob_aux,'artificial'))
    OPT.prob2 = strcat("beats_artificial_train_",int2str(r),".mat");
    DATAtr = data_class_loading(OPT);
    DATAtr = label_encode(DATAtr,OPT);
    OPT.prob2 = strcat("beats_artificial_test_",int2str(r),".mat");
    DATAts = data_class_loading(OPT);
    DATAts = label_encode(DATAts,OPT);
    
elseif (strcmp(prob_aux,'real'))
    OPT.prob2 = strcat("beats_real_train_",int2str(r),".mat");
    DATAtr = data_class_loading(OPT);
    DATAtr = label_encode(DATAtr,OPT);
    OPT.prob2 = strcat("beats_real_test_",int2str(r),".mat");
    DATAts = data_class_loading(OPT);
    DATAts = label_encode(DATAts,OPT);

end

% %%%%%%%%%%%%%%%%% NORMALIZE DATA %%%%%%%%%%%%%%%%%%%%%%%

% Get Normalization Parameters

PARnorm = normalize_fit(DATAtr,OPT);

% Training data normalization

DATAtr = normalize_transform(DATAtr,PARnorm);

% Test data normalization

DATAts = normalize_transform(DATAts,PARnorm);

% Adjust Values for video function

if(strcmp(OPT.prob2,'original'))
    DATA = normalize_transform(DATA,PARnorm);
    DATAtr.Xmax = max(DATA.input,[],2);  % max value
    DATAtr.Xmin = min(DATA.input,[],2);  % min value
    DATAtr.Xmed = mean(DATA.input,2);    % mean value
    DATAtr.Xdp = std(DATA.input,[],2);   % std value
end

% %%%%%%%%%%%%%% SHUFFLE TRAINING DATA %%%%%%%%%%%%%%%%%%%

I = randperm(size(DATAtr.input,2));
DATAtr.input = DATAtr.input(:,I);
DATAtr.output = DATAtr.output(:,I);
DATAtr.lbl = DATAtr.lbl(:,I);

% %%%%%%%%%%% HYPERPARAMETER OPTIMIZATION %%%%%%%%%%%%%%%%

% Using Grid Search or Cross-Validation
% HP = grid_search_cv(DATAtr,HPgs,class_train,class_test,CVp);
% HP = random_search_cv(DATAtr,HPgs,class_train,class_test,CVp);

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

% Generate Report for Test Statistics

class_stats_report(nSTATS_ts);

%% END