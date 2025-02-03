%% Machine Learning ToolBox

% SPOK Model testing in various stationary datasets
% (using hyperparameter optimization)
% Author: David Nascimento Coelho
% Last Update: 2024/05/31

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% Choices

% Datasets Specification

% datasets = 06;      
datasets = [07,10,19,22];

% General options' structure

OPT.Nr = 10;        % Number of experiment realizations
OPT.alg = 'spok';   % Algorithm name
OPT.lbl = 1;        % Type of data labeling. 1: from sequential to [-1 and +1]
OPT.norm = 3;       % Normalization. 0: Don't normalize. 3: z-score norm  
OPT.hold = 1;       % Hold out method.
OPT.ptrn = 0.7;     % Percentage of samples for training. [0,1]

OPT.hpo = 'random'; % 'grid' ; 'random' ; 'none'

% Hyperparameter Optimization (Grid or random search Cross-validation)

CVp.max_it = 100;       % Maximum number of iterations (random search)
CVp.fold = 5;           % number of data partitions for cross validation
CVp.cost = 2;           % 2: Takes into account also the dicitionary size
CVp.lambda = 2;         % Jpbc = Ds + lambda * Err (PB Models)
CVp.gamma = 0.1;        % Jpbc = Ds + lambda * Err + gamma * mcc (PB models)

% Which Kernels Will be tested

% 1: linear | 2: rbf | 3: polynomial | 4: exp | 
% 5: cauchy | 6: log | 7: sigmoid | 8: kmod |

% kernels = 1;
kernels = [1,2,3,4,5,6,7,8];

% Hyperparameters - Default

HP_gs.Ne = 05;                 % Number of epochs
HP_gs.is_static = 1;           % Verify if the dataset is stationary
HP_gs.Dm = 1;                  % Design Method

HP_gs.Ss = 1;                  % Sparsification strategy
HP_gs.v1 = 0.4;                % Sparseness parameter 1 
HP_gs.v2 = 0.9;                % Sparseness parameter 2

HP_gs.Us = 1;                  % Update strategy
HP_gs.eta = 0.1;               % Update rate

HP_gs.Ps = 2;                  % Prunning strategy
HP_gs.min_score = -10;         % Score that leads the sample to be pruned

HP_gs.max_prot = Inf;          % Max number of prototypes
HP_gs.min_prot = 1;            % Min number of prototypes

HP_gs.Von = 0;                 % Enable / disable video 

HP_gs.K = 1;                   % Number of nearest neighbors (classify)
HP_gs.knn_type = 2;            % Type of knn aproximation

HP_gs.Ktype = 2;               % Kernel Type (2: Gaussian / see kernel_func())
HP_gs.sig2n = 0.0001;          % Kernel Regularization parameter
HP_gs.sigma = 2;               % Kernel width (gauss, exp, cauchy, log, kmod)
HP_gs.alpha = 0.1;             % Dot product multiplier (poly 1 / sigm 0.1)
HP_gs.theta = 0.1;             % Dot product adding (poly 1 / sigm 0.1)
HP_gs.gamma = 2;               % polynomial order (poly 2 or 3)

% Obs: the hyperparameters related to kernel functions are at the pipelines

%% Datasets List

% # code: # samples / # attributes / # classes
% Brief Description

% Iris                              => 06: 150 / 04 / 03
% Easiest. Used for debug.

% Motor Failure (prob2 = 1, 2)      => 07: 504 / 06 / 07 or 02
% Short-circuit

% Vertebral Column (prob2 = 1, 2)	=> 10: 310 / 06 / 03 or 02
% Images of Vertebral Columns 
% in order to find deseases

% Cervical Cancer (prob2 = 1, 2)    => 19: 917 / 20 / 07 or 02
% Image of Pap-Smear Cells 
% To detect Cervical Cancer

% Wall-Following                    => 22: 5456 / 02 / 04
% An avoiding Wall Robot.

%% Run algorithm at datasets

for Ss = 1:4
for Dm = 1:2
for K = 1:2
      
HP_gs.Ss = Ss;
HP_gs.Dm = Dm;
if(K == 1)
    HP_gs.K = 1;
else
    HP_gs.K = 2:10;
end

if any(datasets == 06)
    OPT.prob = 06;
    OPT.prob2 = 01;
    exp_spok_stationary_pipeline_1data_1Ss_Nkernel(OPT,HP_gs,CVp,kernels);
end

if any(datasets == 07)
    OPT.prob = 07;
    % OPT.prob2 = 01; % Binary Problem. Unbalanced dataset
    OPT.prob2 = 02; % Binary Problem. Balanced dataset
    exp_spok_stationary_pipeline_1data_1Ss_Nkernel(OPT,HP_gs,CVp,kernels);
end

if any(datasets == 10)
    OPT.prob = 10;
    % OPT.prob2 = 01; % Treated as Multiclass (3 classes)
    OPT.prob2 = 02; % Treated as Binary problem (2 classes)
    exp_spok_stationary_pipeline_1data_1Ss_Nkernel(OPT,HP_gs,CVp,kernels);
end

if any(datasets == 19)
    OPT.prob = 19;
    % OPT.prob2 = 01; % Treated as Multiclass (7 classes)
    OPT.prob2 = 02; % Treated as Binary problem (2 classes)
    exp_spok_stationary_pipeline_1data_1Ss_Nkernel(OPT,HP_gs,CVp,kernels);
end

if any(datasets == 22)
    OPT.prob = 22;
    OPT.prob2 = 01; % with 2 features
    % OPT.prob2 = 02; % with 4 features
    % OPT.prob2 = 03; % with 24 features
    exp_spok_stationary_pipeline_1data_1Ss_Nkernel(OPT,HP_gs,CVp,kernels);
end

end
end
end

%% FINISHED!

% load handel
% sound(y,Fs)

%% END