%% System Identification

% TC1 => Voice Recognition (LMQ, MLP, LPC, PSD, PCA, BC)
% Author: David Nascimento Coelho
% Last Update: 2022/02/24

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% OPTIONS

OPT.prob = 'LPC_F025';	% Choose Dataset 
                    % ('LPC' or 'PSD')
                    % ('PSD_N01','PSD_F01','PSD_N025','PSD_F025')
                    % ('LPC_N01','LPC_F01','LPC_N025','LPC_F025')
                    
OPT.alg = 'MLP';    % Choose algorithm ('MLP' or 'LMQ')

OPT.norm = 3;       % Normalization definition 
                    % (0 - [out=in] | 1 - [0,1] | 2 - [-1,+1] | 3 - zscore)
OPT.lbl = 2;        % Labeling definition
                    % (0 - [out = in] | 1 - [-1 +1]) ; 2 - [0 +1]
                    
OPT.Nr = 500;       % Number of repetitions of the algorithm
OPT.hold = 2;       % Hold out method
OPT.ptrn = 0.5;     % Percentage of samples for training

OPT.bc = 0;         % Apply (or not) Box-Cox Transform
OPT.gamma = 0.1;    % Box-Cox Transform Hyperparameter

OPT.pca = 0;        % Apply (or not) PCA Transform
HPpca.choice = 2;	% Uses beta to define No of attributes
HPpca.beta = 60;    % Number of used attributes
HPpca.tol = 1;      % Tolerance [0 - 1]
HPpca.rem = 1;      % Mean removal [0 or 1] 

%% PRINT OPTIONS

clc;
disp(strcat('Chosen Dataset: ', OPT.prob));
disp(strcat('Chosen Classifier: ', OPT.alg));
disp(strcat('Normalization: ', int2str(OPT.norm)));
disp(strcat('Use Box-Cox: ', int2str(OPT.bc)));
disp(strcat('Use PCA: ', int2str(OPT.pca)));

%% CHOOSE ALGORITHM

% Handlers for classification functions

class_name = OPT.alg;

if(strcmp(OPT.alg,'LMQ'))
    class_train = @ols_train;
    class_test = @ols_classify;

elseif(strcmp(OPT.alg,'MLP'))
    class_train = @mlp_train;
    class_test = @mlp_classify;

end

%% CHOOSE HYPERPARAMETERS

if(strcmp(OPT.alg,'LMQ'))
    HP.aprox = 1;   % type of MQ aproximation (1 -> pinv(X)*y)

elseif(strcmp(OPT.alg,'MLP'))
    HP.Nh = 5;          % Number of hidden neurons
    HP.Ne = 100;       	% maximum number of training epochs
    HP.eta = 0.05;    	% Learning step
    HP.mom = 0.75;    	% Moment Factor
    HP.Nlin = 2;       	% Non-linearity
    HP.Von = 0;         % disable video 

end

%% DATA LOADING

if(strcmp(OPT.prob,'LPC'))
    loadedDATA = load("DATAlpc.mat");
    DATA = loadedDATA.DATAlpc;
    DATA.lbl = DATA.output;

elseif(strcmp(OPT.prob,'PSD'))
    loadedDATA = load("DATApsd.mat");
    DATA = loadedDATA.DATApsd;
    DATA.lbl = DATA.output;
    
elseif(strcmp(OPT.prob,'LPC_N01'))
    loadedDATA = load("DATAlpc_n01.mat");
    DATA = loadedDATA.DATAlpc_n01;
    DATA.lbl = DATA.output;

elseif(strcmp(OPT.prob,'LPC_F01'))
    loadedDATA = load("DATAlpc_f01.mat");
    DATA = loadedDATA.DATAlpc_f01;
    DATA.lbl = DATA.output;

elseif(strcmp(OPT.prob,'LPC_N025'))
    loadedDATA = load("DATAlpc_n025.mat");
    DATA = loadedDATA.DATAlpc_n025;
    DATA.lbl = DATA.output;

elseif(strcmp(OPT.prob,'LPC_F025'))
    loadedDATA = load("DATAlpc_f025.mat");
    DATA = loadedDATA.DATAlpc_f025;
    DATA.lbl = DATA.output;

elseif(strcmp(OPT.prob,'PSD_N01'))
    loadedDATA = load("DATApsd_n01.mat");
    DATA = loadedDATA.DATApsd_n01;
    DATA.lbl = DATA.output;

elseif(strcmp(OPT.prob,'PSD_F01'))
    loadedDATA = load("DATApsd_f01.mat");
    DATA = loadedDATA.DATApsd_f01;
    DATA.lbl = DATA.output;

elseif(strcmp(OPT.prob,'PSD_N025'))
    loadedDATA = load("DATApsd_n025.mat");
    DATA = loadedDATA.DATApsd_n025;
    DATA.lbl = DATA.output;

elseif(strcmp(OPT.prob,'PSD_F025'))
    loadedDATA = load("DATApsd_f025.mat");
    DATA = loadedDATA.DATApsd_f025;
    DATA.lbl = DATA.output;
end

%% DATA PREPROCESSING

DATA = label_encode(DATA,OPT);

if(OPT.bc == 1)
    DATA.input = ((DATA.input).^(OPT.gamma))./(OPT.gamma);
end

%% ACCUMULATORS

NAMES = {'train','test'};           % Acc of names for plots
DATA_acc = cell(OPT.Nr,1);       	% Acc of Data
PAR_acc = cell(OPT.Nr,1);         	% Acc of Parameters and Hyperparameters
STATS_tr_acc = cell(OPT.Nr,1);   	% Acc of Statistics of training data
STATS_ts_acc = cell(OPT.Nr,1);   	% Acc of Statistics of test data
nSTATS_all = cell(2,1);             % Acc of General statistics

%% HOLD OUT / NORMALIZE / SHUFFLE / HPO / TRAINING / TEST / STATISTICS

for r = 1:OPT.Nr
    
% %%%%%%%%%%%%%%%%%%%% HOLD OUT %%%%%%%%%%%%%%%%%%%%%%%%%%

DATA_acc{r} = hold_out(DATA,OPT);   % Hold Out Function
DATAtr = DATA_acc{r}.DATAtr;        % Training Data
DATAts = DATA_acc{r}.DATAts;      	% Test Data

% %%%%%%%%%%%%%%%%% NORMALIZE DATA %%%%%%%%%%%%%%%%%%%%%%%

PARnorm = normalize_fit(DATAtr,OPT);          % Get Norm Parameters
DATAtr = normalize_transform(DATAtr,PARnorm); % Train data normalization
DATAts = normalize_transform(DATAts,PARnorm); % Test data normalization

% %%%%%%%%%%%%%%%%%%%% APPLY PCA %%%%%%%%%%%%%%%%%%%%%%%%%

if(OPT.pca == 1)
    PARpca = pca_feature(DATAtr,HPpca);
    DATAtr = pca_transform(DATAtr,PARpca);
    DATAts = pca_transform(DATAts,PARpca);
end

% disp(size(DATAtr.input));

% %%%%%%%%%%%%%% SHUFFLE TRAINING DATA %%%%%%%%%%%%%%%%%%%

I = randperm(size(DATAtr.input,2));
DATAtr.input = DATAtr.input(:,I);
DATAtr.output = DATAtr.output(:,I);

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

%% RESULTS / STATISTICS

% Statistics for n turns

nSTATS_tr = class_stats_nturns(STATS_tr_acc);
nSTATS_ts = class_stats_nturns(STATS_ts_acc);

% Get all Statistics in one Cell

nSTATS_all{1,1} = nSTATS_tr;
nSTATS_all{2,1} = nSTATS_ts;

% Compare Training and Test Statistics

class_stats_ncomp(nSTATS_all,NAMES); 

%% GENERATE RESULTS

disp('Mean Accuracy:');
disp(nSTATS_ts.acc_mean);
disp('Max Accuracy:');
disp(nSTATS_ts.acc_max);
disp('Min Accuracy:');
disp(nSTATS_ts.acc_min);
disp('Median Accuracy:');
disp(nSTATS_ts.acc_median);
disp('Std Accuracy:');
disp(nSTATS_ts.acc_std);
disp('Max Accuracy index:');
disp(nSTATS_ts.acc_max_i);
disp('Min Accuracy index:');
disp(nSTATS_ts.acc_min_i);


%% END