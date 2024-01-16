%% RESULT ANALYSIS

% KSOM Algorithms and Stationary Data Sets
% Author: David Nascimento Coelho
% Last Update: 2024/01/12

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% MOTOR FAILURE (02 - balanced), KSOMEF, HPO RANDOM, ONE KERNEL

% Just need to modify the number after Kt (from 1 to 8)
results = load('motorFailure_prob2_2_ksomef_hpo_random_norm_3_lbl_1_nn_1_Nep_50_Kt_1.mat');

NAMES = results.NAMES;
disp(NAMES);

%% MOTOR FAILURE (01 - unbalanced), KSOMEF, HPO RANDOM, VARIOUS KERNELS

% Load Results
results = load('motorFailure_prob2_1_ksomef_hpo_random_norm3_1nn.mat');

% Kernel Names
NAMES = {'Linear','Gaussian',... 
         'Polynomial', 'Exponential',...
         'Cauchy', 'Log',...
         'Sigmoid', 'Kmod'};

% Train Stats
class_stats_ncomp(results.variables.nstats_all_tr,NAMES);

nstats_all{1,1} = ksomef_nstats_tr;
nstats_all{2,1} = ksomef_nstats_ts;
NAMES = {'train','test'};
class_stats_ncomp(nstats_all,NAMES);


%% 

NAMES = {'Linear','Gaussian',... 
         'Polynomial', 'Exponential',...
         'Cauchy', 'Log',...
         'Sigmoid', 'Kmod'};

nstats_all_tr = variables.nstats_all_tr;

class_stats_ncomp(nstats_all_tr,NAMES);

nstats_all_ts = variables.nstats_all_ts;

class_stats_ncomp(nstats_all_ts,NAMES);

%% END


















