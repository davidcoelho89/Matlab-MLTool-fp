%% Machine Learning ToolBox

% Classification Algorithms - Unit Test
% Author: David Nascimento Coelho
% Last Update: 2020/05/11

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% GENERAL DEFINITIONS

% General options' structure

OPT.prob = 04;        	% Which problem will be solved / used
OPT.prob2 = 30;       	% More details about a specific data set
OPT.norm = 2;         	% Normalization definition
OPT.lbl = 1;           	% Labeling definition
OPT.Nr = 10;           	% Number of repetitions of the algorithm
OPT.hold = 2;         	% Hold out method
OPT.ptrn = 0.7;        	% Percentage of samples for training
OPT.file = 'fileX.mat';	% file where all the variables will be saved

% Grid Search Parameters

GSp.fold = 5;           % number of data partitions for cross validation
GSp.type = 1;           % Takes into account just accuracy
GSp.lambda = 0.5;       % Jpbc = Ds + lambda * Err (prototype-based models)

%% CHOOSE ALGORITHM

% Handlers for classification functions

class_name = 'MLP';
class_train = @mlp_train;
class_test = @mlp_classify;

%% CHOOSE HYPERPARAMETERS

HP.Nh = 5;          % Number of hidden neurons
HP.Ne = 200;       	% maximum number of training epochs
HP.eta = 0.05;    	% Learning step
HP.mom = 0.75;    	% Moment Factor
HP.Nlin = 2;       	% Non-linearity
HP.Von = 0;         % disable video 

%% HYPERPARAMETERS - FOR OPTIMIZATION

HPgs = HP;

% Can put here vectors of hyperparameters to be optimized
% Ex: HPgs.eta = 0.01:0.01:0.1

%% DATA LOADING, PRE-PROCESSING, VISUALIZATION

DATA = data_class_loading(OPT);     % Load Data Set

DATA = label_encode(DATA,OPT);      % adjust labels for the problem

% plot_data_pairplot(DATA);           % See pairplot of attributes

%% ACCUMULATORS

NAMES = {'train','test'};           % Acc of names for plots
DATA_acc = cell(OPT.Nr,1);       	% Acc of Data
PAR_acc = cell(OPT.Nr,1);         	% Acc of Parameters and Hyperparameters
STATS_tr_acc = cell(OPT.Nr,1);   	% Acc of Statistics of training data
STATS_ts_acc = cell(OPT.Nr,1);   	% Acc of Statistics of test data
nSTATS_all = cell(2,1);             % Acc of General statistics

%% HOLD OUT / NORMALIZE / SHUFFLE / HPO / TRAINING / TEST / STATISTICS

disp('Begin Algorithm');

for r = 1:OPT.Nr

% %%%%%%%%% DISPLAY REPETITION AND DURATION %%%%%%%%%%%%%%

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
HP = grid_search_cv(DATAtr,HPgs,class_train,class_test,GSp);

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

%% GRAPHICS - OF LAST TURN

% % Get Data, Parameters, Statistics
% DATAf.input = [DATAtr.input, DATAts.input];
% DATAf.output = [DATAtr.output, DATAts.output];
% PAR = PAR_acc{r};
% STATS = STATS_ts_acc{r};
% 
% % Classifier Decision Boundaries
% plot_class_boundary(DATA,PAR_acc{r},class_test);
% 
% % ROC Curve (one for each class)
% plot_stats_roc_curve(STATS);
% 
% % Precision-Recall (one for each class)
% plot_stats_precision_recall(STATS)

% % See Class Boundary Video (of last turn)
% if (HP.Von == 1),
%     VID = PAR_acc{r}.VID
%     figure;
%     movie(VID)
% end

%% SAVE VARIABLES AND VIDEO

% % Save All Variables
% save(OPT.file);
% 
% % Save Class Boundary Video (of last turn)
% v = VideoWriter('video.mp4','MPEG-4'); % v = VideoWriter('video.avi');
% v.FrameRate = 1;
% open(v);
% writeVideo(v,VID);
% close(v);

%% END