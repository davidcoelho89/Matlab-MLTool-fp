%% Machine Learning ToolBox

% isk2nn with various streaming datasets
% Author: David Nascimento Coelho
% Last Update: 2021/08/24

%% Choices

% datasets = [28,29,30,33,34,35,36,37,38];
datasets = 28;  % Which dataset(s) will be used
OPT.lbl = 1;    % Type of labeling. 1: from sequential to [-1 and +1]
OPT.norm = 0;   % Normalization. 0: Don't normalize. 3: z-score normalization.
OPT.prob2 = 1;  % Specific choice about dataset

% Grid-Search Cross-validation

PSpar.iterations = 5; % number of times data is presented to the algorithm
PSpar.type = 2;       % 2: Takes into account also the dicitionary size
PSpar.lambda = 2;     % Jpbc = Ds + lambda * Err

% Kernels

% 1: linear | 2: rbf | 3: polynomial | 4: exp | 
% 5: cauchy | 6: log | 7: sigmoid | 8: kmod |
% kernels = [1,2,5];
kernels = 2;

% Hyperparameters - Default

HP_gs.Ne = 01;
HP_gs.Dm = 2;
HP_gs.Ss = 1;
HP_gs.v1 = 0.4;
HP_gs.v2 = 0.9;
HP_gs.Us = 1;
HP_gs.eta = 0.1;
HP_gs.Ps = 2;
HP_gs.min_score = -10;
HP_gs.max_prot = 600;
HP_gs.min_prot = 1;
HP_gs.Von = 0;
HP_gs.K = 1;
HP_gs.knn_type = 2;
HP_gs.Ktype = 1;
HP_gs.sig2n = 0.001;

% Hyperparameter - specific

OPT.max_prot_after_gs = 600;

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

if any(datasets == 25)
    OPT.prob = 25;
    exp_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT,HP_gs,PSpar,kernels);
end

if any(datasets == 26)
    OPT.prob = 26;
    exp_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT,HP_gs,PSpar,kernels);
end

if any(datasets == 27)
    OPT.prob = 27;
    exp_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT,HP_gs,PSpar,kernels);
end

if any(datasets == 28)
    OPT.prob = 28;
    exp_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT,HP_gs,PSpar,kernels);
end

if any(datasets == 29)
    OPT.prob = 29;
    exp_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT,HP_gs,PSpar,kernels);
end

if any(datasets == 30)
    OPT.prob = 30;
    exp_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT,HP_gs,PSpar,kernels);
end

if any(datasets == 31)
    OPT.prob = 31;
    exp_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT,HP_gs,PSpar,kernels);
end

if any(datasets == 32)
    OPT.prob = 32;
    exp_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT,HP_gs,PSpar,kernels);
end

if any(datasets == 33)
    OPT.prob = 33;
    exp_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT,HP_gs,PSpar,kernels);
end

if any(datasets == 34)
    OPT.prob = 34;
    exp_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT,HP_gs,PSpar,kernels);
end

if any(datasets == 35)
    OPT.prob = 35;
    exp_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT,HP_gs,PSpar,kernels);
end

if any(datasets == 36)
    OPT.prob = 36;
    exp_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT,HP_gs,PSpar,kernels);
end

if any(datasets == 37)
    OPT.prob = 37;
    exp_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT,HP_gs,PSpar,kernels);
end

if any(datasets == 38)
    OPT.prob = 38;
    exp_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT,HP_gs,PSpar,kernels);
end

if any(datasets == 39)
    OPT.prob = 39;
    exp_isk2nn_pipeline_streaming_1data_1Ss_Nkernel(OPT,HP_gs,PSpar,kernels);
end

%% FINISHED!

% load handel
% sound(y,Fs)

%% END