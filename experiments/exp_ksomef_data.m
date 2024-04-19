%% Machine Learning ToolBox

% KSOM-EF Model testing in various stationary datasets
% Author: David Nascimento Coelho
% Last Update: 2024/04/18

clear;
clc;
format long e;

%% Choices

% Datasets Specification

datasets = 07;      % datasets = [06,07,10,19,22];

% General options' structure

OPT.lbl = 1;        % Type of data labeling. 1: from sequential to [-1 and +1]
OPT.norm = 3;       % Normalization. 0: Don't normalize. 3: z-score norm 
OPT.Nr = 10;        % Number of experiment realizations
OPT.hold = 01;      % Hold out method
OPT.ptrn = 0.7;     % Percentage of samples for training. [0,1]

OPT.hpo = 'random'; % 'grid' ; 'random' ; 'none'

OPT.savefile = 1;   % decides if file will be saved

OPT.calculate_bin = 0;  % [0 or 1] decides to calculate binary statistics
OPT.class_1_vect = 1;   % [2,3] which classes belongs together
                        % (for binary statistics)

% Metaparameters (Grid or random search Cross-validation)

MP.max_it = 100;   	% Maximum number of iterations (random search)
MP.fold = 5;     	% number of data partitions (cross validation)
MP.cost = 2;        % Takes into account also the dicitionary size
MP.lambda = 2.0;    % Jpbc = Ds + lambda * Err
MP.gamma = 0.1;     % Jpbc = Ds + lambda * Err + gamma * mcc

% Which Kernels Will be tested

% 1: linear | 2: rbf | 3: polynomial | 4: exp | 
% 5: cauchy | 6: log | 7: sigmoid | 8: kmod |

kernels = [1,2,3,4,5,6,7,8];    % kernels = 1;

% Hyperparameters - Default

HP_gs.lbl = 1;          % Neurons' labeling function
HP_gs.Nep = 50;         % max number of epochs
HP_gs.Nk = [6 5];       % number of neurons (prototypes)
HP_gs.init = 02;        % neurons' initialization
HP_gs.dist = 02;        % type of distance
HP_gs.learn = 02;       % type of learning step
HP_gs.No = 0.7;         % initial learning step
HP_gs.Nt = 0.01;        % final learning step
HP_gs.Nn = 01;          % number of neighbors
HP_gs.neig = 03;        % type of neighbor function
HP_gs.Vo = 0.8;         % initial neighbor constant
HP_gs.Vt = 0.3;         % final neighbor constant
HP_gs.Von = 0;          % disable video
HP_gs.K = 1;         	% Number of nearest neighbors (classify)
HP_gs.knn_type = 2; 	% Type of knn aproximation
HP_gs.Ktype = 3;        % Type of Kernel

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

for prot_lbl = 1:3

HP_gs.lbl = prot_lbl;

if any(datasets == 06) % Iris
    OPT.prob = 06;
    OPT.prob2 = 01;
    exp_ksomef_pipeline_1_data_1_lbl_N_kernel(OPT,HP_gs,MP,kernels);
end

if any(datasets == 07) % Motor Failure
    OPT.prob = 07;
    % OPT.prob2 = 01; % Binary Problem. Unbalanced dataset
    OPT.prob2 = 02; % Binary Problem. Balanced dataset
    exp_ksomef_pipeline_1_data_1_lbl_N_kernel(OPT,HP_gs,MP,kernels);
end

if any(datasets == 10) % Vertebral Column
    OPT.prob = 10;
    % OPT.prob2 = 01; % Treated as Multiclass (3 classes)
    OPT.prob2 = 02; % Treated as Binary problem (2 classes)
    exp_ksomef_pipeline_1_data_1_lbl_N_kernel(OPT,HP_gs,MP,kernels);
end

if any(datasets == 19) % Cervical Cancer
    OPT.prob = 19;
    % OPT.prob2 = 01; % Treated as Multiclass (7 classes)
    OPT.prob2 = 02; % Treated as Binary problem (2 classes)
    exp_ksomef_pipeline_1_data_1_lbl_N_kernel(OPT,HP_gs,MP,kernels);
end

if any(datasets == 22) % Wall-Following
    OPT.prob = 22;
    OPT.prob2 = 01; % with 2 features
    % OPT.prob2 = 02; % with 4 features
    % OPT.prob2 = 03; % with 24 features
    exp_ksomef_pipeline_1_data_1_lbl_N_kernel(OPT,HP_gs,MP,kernels);
end

end

%% FINISHED!

% load handel
% sound(y,Fs)

%% END