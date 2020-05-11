%% Machine Learning ToolBox

% Classification Algorithms - Unit Test
% Author: David Nascimento Coelho
% Last Update: 2020/02/02

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% GENERAL DEFINITIONS

% General options' structure

OPT.prob = 11;              % Which problem will be solved / used
OPT.prob2 = 30;             % More details about a specific data set
OPT.norm = 2;               % Normalization definition
OPT.lbl = 1;                % Labeling definition
OPT.Nr = 02;              	% Number of repetitions of the algorithm
OPT.hold = 2;               % Hold out method
OPT.ptrn = 0.7;             % Percentage of samples for training
OPT.file = 'fileX.mat';     % file where all the variables will be saved

% Grid Search Parameters

GSp.fold = 5;       % number of data partitions for cross validation
GSp.type = 1;       % Takes into account just accuracy
GSp.lambda = 0.5; 	% Jpbc = Ds + lambda * Err (prototype-based models)

%% CHOOSE ALGORITHM

% Handlers for classification functions

class_name = 'isk2nn';
class_train = @isk2nn_train;
class_test = @isk2nn_classify;

%% CHOOSE HYPERPARAMETERS

HP.Dm = 2;          % Design Method
HP.Ss = 1;          % Sparsification strategy
HP.v1 = 0.8;        % Sparseness parameter 1 
HP.v2 = 0.9;        % Sparseness parameter 2
HP.Us = 1;          % Update strategy
HP.eta = 0.01;      % Update rate
HP.Ps = 1;          % Prunning strategy
HP.min_score = -10; % Score that leads the sample to be pruned
HP.max_prot = Inf;  % Max number of prototypes
HP.min_prot = 1;    % Min number of prototypes
HP.Von = 0;         % Enable / disable video 
HP.K = 1;           % Number of nearest neighbors (classify)
HP.knn_type = 2;    % Type of knn aproximation
HP.Ktype = 2;       % Kernel Type
HP.sig2n = 0.001;   % Kernel Regularization parameter
HP.sigma = 2;    	% Kernel width (gauss, exp, cauchy, log, kmod)
HP.gamma = 2;       % polynomial order (poly 2 or 3)
HP.alpha = 0.1;     % Dot product multiplier (poly 1 / sigm 0.1)
HP.theta = 0.1;     % Dot product adding (poly 1 / sigm 0.1)

%% HYPERPARAMETERS - FOR OPTIMIZATION

HPgs = HP;

% Can put here vectors of hyperparameters to be optimized
% Ex: HPgs.v1 = [0.2, 0.4, 0.6, 0.8, 1.0]

%% DATA LOADING, PRE-PROCESSING, VISUALIZATION

DATA = data_class_loading(OPT);     % Load Data Set

DATA = label_encode(DATA,OPT);      % adjust labels for the problem

plot_data_pairplot(DATA);           % See pairplot of attributes

%% ACCUMULATORS

NAMES = {'train','test'};           % Acc of names for plots
DATA_acc = cell(OPT.Nr,1);       	% Acc of Data
PAR_acc = cell(OPT.Nr,1);         	% Acc of Parameters and Hyperparameters
STATS_tr_acc = cell(OPT.Nr,1);   	% Acc of Statistics of training data
STATS_ts_acc = cell(OPT.Nr,1);   	% Acc of Statistics of test data
nSTATS_all = cell(2,1);             % Acc of General statistics

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
HP = grid_search_cv(DATAtr,HPgs,class_train,class_test,GSp);

% %%%%%%%%%%%%%% CLASSIFIER'S TRAINING %%%%%%%%%%%%%%%%%%%

PAR_acc{r} = class_train(DATAtr,HP);	% Calculate parameters

% %%%%%%%%% CLASSIFIER'S TEST AND STATISTICS %%%%%%%%%%%%%

OUTtr = class_test(DATAtr,PAR_acc{r});          	% Outputs with training data
STATS_tr_acc{r} = class_stats_1turn(DATAtr,OUTtr);  % Results with training data

OUTts = class_test(DATAts,PAR_acc{r});          	% Outputs with test data
STATS_ts_acc{r} = class_stats_1turn(DATAts,OUTts);  % Results with test data

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

% Boxplot of Training and Test Accuracies

class_stats_ncomp(nSTATS_all,NAMES); 

%% GRAPHICS - OF LAST TURN

% Get Data
DATAf.input = [DATAtr.input, DATAts.input];
DATAf.output = [DATAtr.output, DATAts.output];

% Classifier Decision Boundaries
plot_class_boundary(DATAf,PAR_acc{r},class_test);

% ROC Curve (one for each class)
plot_stats_roc_curve(STATS_ts_acc{r});

% Precision-Recall (one for each class
plot_stats_precision_recall(STATS_ts_acc{r})

%% SAVE VARIABLES AND VIDEO

% % Save All Variables
% save(OPT.file);
% 
% % See Class Boundary Video (of last turn)
% if (HP.Von == 1),
%     figure;
%     movie(PAR_acc{r}.VID)
% end
% 
% % Save Class Boundary Video (of last turn)
% v = VideoWriter('video.mp4','MPEG-4'); % v = VideoWriter('video.avi');
% v.FrameRate = 1;
% open(v);
% writeVideo(v,PAR_acc{r}.VID);
% close(v);

%% END