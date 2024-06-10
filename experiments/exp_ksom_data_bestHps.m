%% Machine Learning ToolBox

% KSOM Model testing in various stationary datasets
% (using the best hyperparameters found using hpo)
% Author: David Nascimento Coelho
% Last Update: 2024/06/10

clear;
clc;
format long e;

%% Choices

% Datasets Specification

datasets = 07;      % datasets = [06,07,10,19,22];

% General options' structure

OPT.Nr = 100;       % Number of experiment realizations
% OPT.alg = 'ksom_ef';% ksom_ef or ksom_gd
OPT.lbl = 1;        % Type of data labeling. 1: from sequential to [-1 and +1]
OPT.norm = 3;       % Normalization. 0: Don't normalize. 3: z-score norm 
OPT.hold = 1;       % Hold out method
OPT.ptrn = 0.7;     % Percentage of samples for training. [0,1]
OPT.hpo = 'best';   % 'grid' ; 'random' ; 'none' ; 'best'
OPT.savefile = 1;   % decides if file will be saved

% OPT.calculate_bin = 0;  % [0 or 1] decides to calculate binary statistics
% OPT.class_1_vect = 1;   % [2,3] which classes belongs together
%                         % (for binary statistics)

% Which Kernels Will be tested

% 1: linear | 2: rbf | 3: polynomial | 4: exp | 
% 5: cauchy | 6: log | 7: sigmoid | 8: kmod |

% kernels = 1;
kernels = [1,2,3,4,5,6,7,8];

% Specific Hyperparameters

OPT.nn = '1';           % Number of Nearest Neighbors: 1 '1' ; or >1 '2'
OPT.Nep = '50';         % Number of epochs
OPT.Nprot = '30';       % Number of prototypes

algs = {'ksom_ef','ksom_gd'};   % Which Algorithms
prot_lbls = [1,2,3];            % Which labeling strategies

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

for i = 1:length(algs)
for j = 1:length(prot_lbls)

OPT.alg = algs{i};
OPT.prot_lbl = prot_lbls(j);

if any(datasets == 06) % Iris
    OPT.prob = 06;
    OPT.prob2 = 01;
    exp_ksom_pipeline_1_data_1_lbl_N_kernel_best(OPT,kernels);
end

if any(datasets == 07) % Motor Failure
    OPT.prob = 07;
    % OPT.prob2 = 01; % Binary Problem. Unbalanced dataset
    OPT.prob2 = 02; % Binary Problem. Balanced dataset
    exp_ksom_pipeline_1_data_1_lbl_N_kernel_best(OPT,kernels);
end

if any(datasets == 10) % Vertebral Column
    OPT.prob = 10;
    % OPT.prob2 = 01; % Treated as Multiclass (3 classes)
    OPT.prob2 = 02; % Treated as Binary problem (2 classes)
    exp_ksom_pipeline_1_data_1_lbl_N_kernel_best(OPT,kernels);
end

if any(datasets == 19) % Cervical Cancer
    OPT.prob = 19;
    % OPT.prob2 = 01; % Treated as Multiclass (7 classes)
    OPT.prob2 = 02; % Treated as Binary problem (2 classes)
    exp_ksom_pipeline_1_data_1_lbl_N_kernel_best(OPT,kernels);
end

if any(datasets == 22) % Wall-Following
    OPT.prob = 22;
    OPT.prob2 = 01; % with 2 features
    % OPT.prob2 = 02; % with 4 features
    % OPT.prob2 = 03; % with 24 features
    exp_ksom_pipeline_1_data_1_lbl_N_kernel_best(OPT,kernels);
end

end
end

%% FINISHED!

% load handel
% sound(y,Fs)

%% END