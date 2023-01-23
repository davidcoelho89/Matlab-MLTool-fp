%% Machine Learning ToolBox

% Spok Model testing in various stationary datasets
% Author: David Nascimento Coelho
% Last Update: 2023/01/14

%% Choices

% Datasets Specification

datasets = 06;  % datasets = [06,07,10,19,22];

OPT.prob2 = 1;  % Some especific characteristic of a dataset
OPT.lbl = 1;    % Type of labeling. 1: from sequential to [-1 and +1]
OPT.norm = 3;   % Normalization. 0: Don't normalize. 3: z-score norm  
OPT.Nr = 10;    % Number of repetitions
OPT.hold = 2;   % Hold out method.
OPT.ptrn = 0.7; % Percentage of samples for training. [0,1]

% Hyperparameter Optimization (Grid or random search Cross-validation)

CVp.max_it = 9;         % Maximum number of iterations (random search)
CVp.fold = 5;           % number of data partitions for cross validation
CVp.cost = 2;           % Which cost function will be used
CVp.lambda = 2;         % Jpbc = Ds + lambda * Err (prototype-based models)

% Which Kernels Will be tested

% 1: linear | 2: rbf | 3: polynomial | 4: exp | 
% 5: cauchy | 6: log | 7: sigmoid | 8: kmod |

kernels = 2;            % kernels = [1,2,3,4,5,6,7,8];

% Hyperparameters - Default

HP_gs.Ne = 01;                 % Number of epochs
HP_gs.Dm = 2;                  % Design Method
HP_gs.Ss = 1;                  % Sparsification strategy
HP_gs.v1 = 0.4;                % Sparseness parameter 1 
HP_gs.v2 = 0.9;                % Sparseness parameter 2
HP_gs.Us = 1;                  % Update strategy
HP_gs.eta = 0.01;              % Update rate
HP_gs.Ps = 2;                  % Prunning strategy
HP_gs.min_score = -10;         % Score that leads the sample to be pruned
HP_gs.max_prot = 600;          % Max number of prototypes
HP_gs.min_prot = 1;            % Min number of prototypes
HP_gs.Von = 0;                 % Enable / disable video 
HP_gs.K = 1;                   % Number of nearest neighbors (classify)
HP_gs.knn_type = 2;            % Type of knn aproximation
HP_gs.Ktype = 2;               % Kernel Type (2: Gaussian / see kernel_func())
HP_gs.sig2n = 0.001;           % Kernel Regularization parameter
HP_gs.sigma = 2;               % Kernel width (gauss, exp, cauchy, log, kmod)
HP_gs.alpha = 0.1;             % Dot product multiplier (poly 1 / sigm 0.1)
HP_gs.theta = 0.1;             % Dot product adding (poly 1 / sigm 0.1)
HP_gs.gamma = 2;               % polynomial order (poly 2 or 3)

%% Datasets List

% # code: # samples / # attributes / # classes
% Brief Description

% Iris                          => 06: 150 / 04 / 03
% Easiest. Used for debug.

% Motor Failure                 => 07: 504 / 06 / 02
% Short-circuit

% Vertebral Column              => 10: xx / xx / xx
% Images of Vertebral Columns 
% in order to find deseases

% Cervical Cancer               => 19: 917 / 20 / 02
% Image of Pap-Smear Cells 
% To detect Cervical Cancer

% Wall-Following                => 22: 5456 / 02 / 02
% An avoiding Wall Robot.

%% Run algorithm at datasets

if any(datasets == 06)
    OPT.prob = 06;
    exp_spok_stationary_pipeline_1data_1Ss_Nkernel(OPT,HP_gs,CVp,kernels);
end

if any(datasets == 07)
    OPT.prob = 07;
    exp_spok_stationary_pipeline_1data_1Ss_Nkernel(OPT,HP_gs,CVp,kernels);
end

if any(datasets == 10)
    OPT.prob = 10;
    exp_spok_stationary_pipeline_1data_1Ss_Nkernel(OPT,HP_gs,CVp,kernels);
end

if any(datasets == 19)
    OPT.prob = 19;
    exp_spok_stationary_pipeline_1data_1Ss_Nkernel(OPT,HP_gs,CVp,kernels);
end

if any(datasets == 22)
    OPT.prob = 22;
    exp_spok_stationary_pipeline_1data_1Ss_Nkernel(OPT,HP_gs,CVp,kernels);
end

%% FINISHED!

% load handel
% sound(y,Fs)

%% END