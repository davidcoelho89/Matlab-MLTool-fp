%% MACHINE LEARNING TOOLBOX

% Spok With one stationary Dataset
% Author: David Nascimento Coelho
% Last Update: 2023/01/23

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window
format long e;  % Output data style (float)

%% CHOOSE EXPERIMENT PARAMETERS

% General options' structure

OPT.Nr = 10;            % Number of repetitions of the algorithm
OPT.alg = 'spok';	    % Which classifier will be used
OPT.prob = 06;          % Which problem will be solved / used
OPT.prob2 = 02;         % More details about a specific data set
OPT.norm = 3;           % Normalization definition
OPT.lbl = 1;            % Labeling definition. 1: [-1 +1] pattern
OPT.hold = 2;           % Hold out method
OPT.ptrn = 0.7;         % Percentage of samples for training
OPT.hpo = 'none';       % 'grid' ; 'random' ; 'none'

OPT.savefile = 0;               % decides if file will be saved
OPT.savevideo = 0;              % decides if video will be saved
OPT.show_specific_stats = 0;    % roc, class boundary, precision-recall
OPT.result_analysis = 1;        % show result analysis

% Metaparameters

MP.max_it = 09;   	% Maximum number of iterations (random search)
MP.fold = 5;     	% number of data partitions (cross validation)
MP.cost = 2;        % Takes into account also the dicitionary size
MP.lambda = 0.5; 	% Jpbc = Ds + lambda * Err

%% CHOOSE FIXED HYPERPARAMETERS 

HP.Ne = 01;             	% Maximum number of epochs
HP.is_static = 1;           % Verify if the dataset is stationary
HP.Dm = 2;                  % Design Method
HP.Ss = 1;                  % Sparsification strategy
HP.v1 = 0.1;                % Sparseness parameter 1 
HP.v2 = 0.9;                % Sparseness parameter 2
HP.Us = 0;                  % Update strategy
HP.eta = 0.1;               % Update rate
HP.Ps = 0;                  % Prunning strategy
HP.min_score = -10;         % Score that leads the sample to be pruned
HP.max_prot = Inf;          % Max number of prototypes
HP.min_prot = 1;            % Min number of prototypes
HP.Von = 0;                 % Enable / disable video 
HP.K = 1;                   % Number of nearest neighbors (classify)
HP.knn_type = 2;            % Type of knn aproximation
HP.Ktype = 1;               % Kernel Type (2: Gaussian / see kernel_func())
HP.sig2n = 0.001;           % Kernel Regularization parameter
HP.sigma = 2;               % Kernel width (gauss, exp, cauchy, log, kmod)
HP.alpha = 0.1;             % Dot product multiplier (poly 1 / sigm 0.1)
HP.theta = 0.1;             % Dot product adding (poly 1 / sigm 0.1)
HP.gamma = 2;               % polynomial order (poly 2 or 3)

%% HYPERPARAMETERS - FOR OPTIMIZATION

if(~strcmp(OPT.hpo,'none'))

% Get Default Hyperparameters

HPgs = HP;

% Get specific Hyperparameters

% ALD
if(HP.Ss == 1)
    
    if HP.Ktype == 1
        % Linear
        HPgs.v1 = 2.^linspace(-10,3,14);
        HPgs.v2 = HPgs.v1(end) + 0.001;
    elseif HP.Ktype == 2
        % Gaussian
        HPgs.v1 = 2.^linspace(-4,3,8);
        HPgs.v2 = HPgs.v1(end) + 0.001;
        HPgs.sigma = 2.^linspace(-8,5,14);
    elseif HP.Ktype == 3
        % Polynomial
        HP_gs.v1 = 2.^linspace(-13,6,20);
        HP_gs.v2 = HP_gs.v1(end) + 0.001;
        HP_gs.gamma = [0.2,0.4,0.6,0.8,1,2,2.2,2.4,2.6,2.8,3];
    elseif HP.Ktype == 4
        % Exponential
        HP_gs.v1 = 2.^linspace(-4,3,8);
        HP_gs.v2 = HP_gs.v1(end) + 0.001;
        HPgs.sigma = 2.^linspace(-8,5,14);
    elseif HP.Ktype == 5
        % Cauchy
        HPgs.v1 = 2.^linspace(-4,3,8);
        HPgs.v2 = HPgs.v1(end) + 0.001;
        HPgs.sigma = 2.^linspace(-8,5,14);
    elseif HP.Ktype == 6
        % Log
        HP_gs.v1 = -2.^linspace(10,2,9);
        HP_gs.v2 = HP_gs.v1(end) + 0.001;
        HP_gs.sigma = [0.001 0.01 0.1 1 2 5];
    elseif HP.Ktype == 7
        % Sigmoid
        HP_gs.v1 = 2.^linspace(-13,6,20);
        HP_gs.v2 = HP_gs.v1(end) + 0.001;
        HP_gs.alpha = 2.^linspace(-8,2,11);
    elseif HP.Ktype == 8
        % Kmod
        HP_gs.v1 = 2.^linspace(-13,6,20);
        HP_gs.v2 = HP_gs.v1(end) + 0.001;
        HP_gs.sigma = 2.^linspace(-8,2,11);
        HP_gs.gamma = 2.^linspace(-8,2,11);
    end

% COHERENCE
elseif(HP.Ss == 2)
    
    if HP.Ktype == 1
        % Linear
        HPgs.v1 = [0.001 0.01 0.1 0.3 0.5 0.7 0.9 0.99];
        HPgs.v2 = HPgs.v1(end) + 0.001;
    elseif HP.Ktype == 2
        % Gaussian
        HPgs.v1 = [0.001 0.01 0.1 0.3 0.5 0.7 0.9 0.99];
        HPgs.v2 = HPgs.v1(end) + 0.001;
        HPgs.sigma = 2.^linspace(-10,9,20);
    end

% NOVELTY
elseif(HP.Ss == 3)
    
    
    
% SURPRISE
elseif(HP.Ss == 4)
    
    

end

end

%% DATA LOADING, PRE-PROCESSING, VISUALIZATION

DATA = data_class_loading(OPT);     % Load Data Set

DATA = label_encode(DATA,OPT);      % adjust labels for the problem

% plot_data_pairplot(DATA);           % See pairplot of attributes

%% ACCUMULATORS AND HANDLERS

NAMES = {'train','test'};           % Acc of names for plots
DATA_acc = cell(OPT.Nr,1);       	% Acc of Data
PAR_acc = cell(OPT.Nr,1);         	% Acc of Parameters and Hyperparameters
STATS_tr_acc = cell(OPT.Nr,1);   	% Acc of Statistics of training data
STATS_ts_acc = cell(OPT.Nr,1);   	% Acc of Statistics of test data
nSTATS_all = cell(2,1);             % Acc of General statistics

algorithm_name = upper(OPT.alg);

train_string = strcat(OPT.alg,'_train');
model_train = str2func(train_string);

test_string = strcat(OPT.alg,'_classify');
model_classify = str2func(test_string);

%% VIDEO NAME AND FILE NAME

OPT.filename = strcat(DATA.name,'_',OPT.alg,'_hpo',int2str(OPT.hpo),...
                      '_norm',int2str(OPT.norm),'_Dm',int2str(HP.Dm),...
                      '_Ss',int2str(HP.Ss),'_Us',int2str(HP.Us),...
                      '_Ps',int2str(HP.Ps),'_kernel',int2str(HP.Ktype),...
                      '_',int2str(HP.K),'nn');

OPT.videoname = strcat(OPT.alg,'_',DATA.name,'.mp4');

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

% Update maximum number of prototypes 
% (avoid PBC getting  half of data training points)

Ntr_samples = length(DATAtr.lbl);
HP.max_prot = Ntr_samples / 2;
HPgs.max_prot = Ntr_samples / 2;

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

if(strcmp(OPT.hpo,'none'))
    % Does nothing
elseif(strcmp(OPT.hpo,'grid'))
    HP = grid_search_cv(DATAtr,HPgs,model_train,model_classify,MP);
elseif(strcmp(OPT.hpo,'random'))
    HP = random_search_cv(DATAtr,HPgs,model_train,model_classify,MP);
end

% %%%%%%%%%%%%%% CLASSIFIER'S TRAINING %%%%%%%%%%%%%%%%%%%

% Save video of last training
if ((r == OPT.Nr) && (OPT.savevideo == 1))
    HP.Von = 1;
end

% Calculate model's parameters
PAR_acc{r} = spok_train(DATAtr,HP);

% %%%%%%%%% CLASSIFIER'S TEST AND STATISTICS %%%%%%%%%%%%%

% Results and Statistics with training data
OUTtr = spok_classify(DATAtr,PAR_acc{r});
STATS_tr_acc{r} = class_stats_1turn(DATAtr,OUTtr);

% Results and Statistics with test data
OUTts = spok_classify(DATAts,PAR_acc{r});
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

if(OPT.show_specific_stats == 1)
    
    % Get Data, Parameters, Statistics
    DATAf.input = [DATAtr.input, DATAts.input];
    DATAf.output = [DATAtr.output, DATAts.output];
    PAR = PAR_acc{r};
    STATS = STATS_ts_acc{r};
    
    % Classifier Decision Boundaries
    plot_class_boundary(DATAf,PAR,model_classify);
    
    % ROC Curve (one for each class)
    plot_stats_roc_curve(STATS);
    
    % Precision-Recall (one for each class)
    plot_stats_precision_recall(STATS)

end

% See Class Boundary Video (of last turn)
if (HP.Von == 1)
    VID = PAR_acc{r}.VID;
    figure;
    movie(VID)
end

if (OPT.result_analysis == 1)
    spoknn_stationary_results_analysis;
end

%% SAVE VARIABLES AND VIDEO

% % Save All Variables

if(OPT.savefile)
    save(OPT.filename);
end

% Save Class Boundary Video (of last turn)
if (HP.Von == 1 && OPT.savevideo == 1)
    v = VideoWriter('video.mp4','MPEG-4');
    v.FrameRate = 1;
    VID = PAR_acc{r}.VID;
    open(v);
    writeVideo(v,VID);
    close(v);
end

%% END