%% MACHINE LEARNING TOOLBOX

% Classification Algorithms - Unit Test
% Author: David Nascimento Coelho
% Last Update: 2022/04/11

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window
format long e;  % Output data style (float)

%% CHOOSE EXPERIMENT PARAMETERS

% General options' structure

OPT.Nr = 10;           	% Number of realizations
OPT.alg = 'ols';        % Which classifier will be used
OPT.prob = 06;        	% Which problem will be solved / used
OPT.prob2 = 30;       	% More details about a specific data set
OPT.norm = 3;         	% Normalization definition
OPT.lbl = 1;           	% Labeling definition
OPT.hold = 2;         	% Hold out method
OPT.ptrn = 0.7;        	% Percentage of samples for training
OPT.file = 'fileX.mat';	% file where all the variables will be saved
OPT.hpo = 'none';       % 'grid' ; 'random' ; 'none'

% "Hyperparameters Optimization" Parameters

CVp.max_it = 9;         % Maximum number of iterations (random search)
CVp.fold = 5;           % number of data partitions for cross validation
CVp.cost = 1;           % Which cost function will be used
CVp.lambda = 0.5;       % Jpbc = Ds + lambda * Err (prototype-based models)

%% CHOOSE FIXED HYPERPARAMETERS 

if(strcmp(OPT.alg,'mlp'))
    HP.Nh = 05;         % Number of hidden neurons
    HP.Ne = 200;        % maximum number of training epochs
    HP.eta = 0.05;      % Learning step
    HP.mom = 0.75;      % Moment Factor
    HP.Nlin = 2;        % Non-linearity
    HP.Von = 0;         % disable video
elseif(strcmp(OPT.alg,'lms'))
    HP.Ne = 200;       	% maximum number of training epochs
    HP.eta = 0.05;    	% Learning step
    HP.Von = 0;         % disable video
elseif(strcmp(OPT.alg,'ols'))
    HP.aprox = 1;       % type of approximation
end

%% CHOOSE HYPERPARAMETERS TO BE OPTIMIZED

if(~strcmp(OPT.hpo,'none'))
    
    % Get Default Hyperparameters
    HPgs = HP;

    % Get specific Hyperparameters
    
    if(strcmp(OPT.alg,'mlp'))
        HPgs.Nh = {5,10,20,[2,3],[3,3],[4,5]};
        HPgs.eta = [0.01,0.02,0.03,0.04,0.05,0.1];
        HPgs.Nh = {10,[3,3],[4,5]};
        HPgs.eta = [0.01,0.05,0.1];
    end

end

%% ACCUMULATORS

NAMES = {'train','test'};           % Acc of names for plots
DATA_acc = cell(OPT.Nr,1);       	% Acc of Data
PAR_acc = cell(OPT.Nr,1);         	% Acc of Parameters and Hyperparameters
STATS_tr_acc = cell(OPT.Nr,1);   	% Acc of Statistics of training data
STATS_ts_acc = cell(OPT.Nr,1);   	% Acc of Statistics of test data
nSTATS_all = cell(2,1);             % Acc of General statistics

%% HANDLERS FOR CLASSIFICATION FUNCTIONS

algorithm_name = upper(OPT.alg);

str_train = strcat(lower(OPT.alg),'_train');
class_train = str2func(str_train);

str_test = strcat(lower(OPT.alg),'_classify');
class_test = str2func(str_test);

%% DATA LOADING, PRE-PROCESSING, VISUALIZATION

DATA = data_class_loading(OPT);     % Load Data Set

DATA = label_encode(DATA,OPT);      % adjust labels for the problem

% plot_data_pairplot(DATA);         % See pairplot of attributes

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

% Get Normalization Parameters (from trainig data set)

PARnorm = normalize_fit(DATAtr,OPT);

% Training and Test data normalization

DATAtr = normalize_transform(DATAtr,PARnorm);
DATAts = normalize_transform(DATAts,PARnorm);
DATA = normalize_transform(DATA,PARnorm);

% Adjust Values for video function

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

if(strcmp(OPT.hpo,'none'))
    % Does nothing
elseif(strcmp(OPT.hpo,'grid'))
    HP = grid_search_cv(DATAtr,HPgs,class_train,class_test,CVp);
elseif(strcmp(OPT.hpo,'random'))
    HP = random_search_cv(DATAtr,HPgs,class_train,class_test,CVp);
end

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