%% Machine Learning ToolBox

% Spok Model testing in various streaming datasets
% (using hyperparameter optimization)
% Author: David Nascimento Coelho
% Last Update: 2024/06/15

clear;
clc;
format long e;

format long e;  % Output data style (float)

%% Choices

% Datasets Specification

datasets = 34;  % datasets = [28,29,30,33,34,35,36,37,38]

% General options' structure

OPT.Nr = 1;         % Just need one realization
OPT.alg = 'spok';   % algorithm name
OPT.lbl = 1;        % Type of labeling. 1: from sequential to [-1 and +1]
OPT.norm = 0;       % Normalization. 0: Don't normalize. 3: z-score norm
OPT.hold = 'ttt';   % Test than train
OPT.max_prot_after_gs = 1000;   % max #prototypes after grid-search

OPT.hpo = 'random'; % 'grid' ; 'random' ; 'none'

% Hyperparameter Optimization (Grid or random search Cross-validation)

PSpar.iterations = 5; % Number of times data is presented to the algorithm
PSpar.type = 2;       % 2: Takes into account also the dicitionary size
PSpar.lambda = 2;     % Jpbc = Ds + lambda * Err
PSpar.gamma = 0.1;    % Jpbc = Ds + lambda * Err + gamma * mcc 

% Which Kernels Will be tested

% 1: linear | 2: rbf | 3: polynomial | 4: exp | 
% 5: cauchy | 6: log | 7: sigmoid | 8: kmod |

% kernels = 2;
kernels = [1,2,3,4,5,6,7,8];

% Hyperparameters - Default

HP_gs.Ne = 01;                 % Number of epochs
HP_gs.is_static = 0;           % Verify if the dataset is stationary
HP_gs.Dm = 2;                  % Design Method

HP_gs.Ss = 1;                  % Sparsification strategy
HP_gs.v1 = 0.4;                % Sparseness parameter 1 
HP_gs.v2 = 0.9;                % Sparseness parameter 2

HP_gs.Us = 1;                  % Update strategy
HP_gs.eta = 0.10;              % Update rate

HP_gs.Ps = 2;                  % Prunning strategy
HP_gs.min_score = -10;         % Score that leads the sample to be pruned

HP_gs.max_prot = 600;          % Max number of prototypes
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

% Sea Concepts              => 25: 200k / 03 / 02
% label noise (10%)
% f1 + f2 = b; b is changing each 5000 samples.
% Abrupt drift

% Rotating Hyperplane       => 26: 200k / 10 / 02. 
% Moving Hyperplane. 
% Gradual Drift.

% RBF Moving                => 27: 200k / 10 / 05. 
% Moving RBFs. Different Mean. 
% Gradual drift.

% RBF Interchange           => 28: 200k / 02 / 15. 
% Interchanging RBFs. Change Means. Abrupt drift.

% Moving Squares            => 29: 200k / 02 / 04. 
% Moving Squares. Gradual/Incremental drift.

% Transient Chessboard      => 30: 200k / 02 / 08. 
% Virtual Reocurring drifts.

% Mixed Drift               => 31: 600k / 02 / 15. 
% Various drifts.

% LED                       => 32: 200k / 24 / 10
% Atributes = 0 or 1. Represents a 7 segments display.
% 17 Irrelevant Attributes. Which attribute is irrelevant: changes.
% Incremental Drift.

% Weather                   => 33: 18159 / 08 / 02
% Virtual Drift

% Electricity               => 34: 45312 / 08 / 02
% Real Drift

% Cover Type                => 35: 581012 / 54 / 07
% Real Drift

% Poker Hand                => 36: 829201 / 10 / 10
% Virtual Drift

% Outdoor                   => 37: 4000 / 21 / 40
% Virtual Drift

% Rialto                    => 38: 82250 / 27 / 10
% Virtual Drift

% Spam                      => 39: 
% Real Drift

%% Run algorithm at datasets

for Ss = 1:4
for K = 1:2

HP_gs.Ss = Ss;
if(K == 1)
    HP_gs.K = 1;
else
    HP_gs.K = 2:10;
end

if any(datasets == 25)
    OPT.prob = 25;
    OPT.prob2 = 1;  % Specific choice about dataset
    exp_spok_streaming_pipeline_1data_1Ss_Nkernel(OPT,HP_gs,PSpar,kernels);
end

if any(datasets == 26)
    OPT.prob = 26;
    OPT.prob2 = 1;  % Specific choice about dataset
    exp_spok_streaming_pipeline_1data_1Ss_Nkernel(OPT,HP_gs,PSpar,kernels);
end

if any(datasets == 27)
    OPT.prob = 27;
    OPT.prob2 = 1;  % Specific choice about dataset
    exp_spok_streaming_pipeline_1data_1Ss_Nkernel(OPT,HP_gs,PSpar,kernels);
end

if any(datasets == 28)
    OPT.prob = 28;
    OPT.prob2 = 1;  % Specific choice about dataset
    exp_spok_streaming_pipeline_1data_1Ss_Nkernel(OPT,HP_gs,PSpar,kernels);
end

if any(datasets == 29)
    OPT.prob = 29;
    OPT.prob2 = 1;  % Specific choice about dataset
    exp_spok_streaming_pipeline_1data_1Ss_Nkernel(OPT,HP_gs,PSpar,kernels);
end

if any(datasets == 30)
    OPT.prob = 30;
    OPT.prob2 = 1;  % Specific choice about dataset
    exp_spok_streaming_pipeline_1data_1Ss_Nkernel(OPT,HP_gs,PSpar,kernels);
end

if any(datasets == 31)
    OPT.prob = 31;
    OPT.prob2 = 1;  % Specific choice about dataset
    exp_spok_streaming_pipeline_1data_1Ss_Nkernel(OPT,HP_gs,PSpar,kernels);
end

if any(datasets == 32)
    OPT.prob = 32;
    OPT.prob2 = 1;  % Specific choice about dataset
    exp_spok_streaming_pipeline_1data_1Ss_Nkernel(OPT,HP_gs,PSpar,kernels);
end

if any(datasets == 33)
    OPT.prob = 33;
    OPT.prob2 = 1;  % Specific choice about dataset
    exp_spok_streaming_pipeline_1data_1Ss_Nkernel(OPT,HP_gs,PSpar,kernels);
end

if any(datasets == 34)
    OPT.prob = 34;
    OPT.prob2 = 1;  % Specific choice about dataset
    exp_spok_streaming_pipeline_1data_1Ss_Nkernel(OPT,HP_gs,PSpar,kernels);
end

if any(datasets == 35)
    OPT.prob = 35;
    OPT.prob2 = 1;  % Specific choice about dataset
    exp_spok_streaming_pipeline_1data_1Ss_Nkernel(OPT,HP_gs,PSpar,kernels);
end

if any(datasets == 36)
    OPT.prob = 36;
    OPT.prob2 = 1;  % Specific choice about dataset
    exp_spok_streaming_pipeline_1data_1Ss_Nkernel(OPT,HP_gs,PSpar,kernels);
end

if any(datasets == 37)
    OPT.prob = 37;
    OPT.prob2 = 1;  % Specific choice about dataset
    exp_spok_streaming_pipeline_1data_1Ss_Nkernel(OPT,HP_gs,PSpar,kernels);
end

if any(datasets == 38)
    OPT.prob = 38;
    OPT.prob2 = 1;  % Specific choice about dataset
    exp_spok_streaming_pipeline_1data_1Ss_Nkernel(OPT,HP_gs,PSpar,kernels);
end

if any(datasets == 39)
    OPT.prob = 39;
    OPT.prob2 = 1;  % Specific choice about dataset
    exp_spok_streaming_pipeline_1data_1Ss_Nkernel(OPT,HP_gs,PSpar,kernels);
end

end
end

%% FINISHED!

% load handel
% sound(y,Fs)

%% END