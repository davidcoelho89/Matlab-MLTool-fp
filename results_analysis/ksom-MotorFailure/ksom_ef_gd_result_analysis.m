%% RESULT ANALYSIS

% KSOM Algorithms and Stationary Data Sets
% Author: David Nascimento Coelho
% Last Update: 2024/01/12

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% LOAD RESULTS

results01 = load('motorFailure_ksomef_hpo_none_norm3_1nn.mat');
results02 = load('motorFailure_ksomgd_hpo_none_norm0_1nn.mat');
results03 = load('motorFailure_ksomgd_hpo_random_norm0_1nn.mat');
results04 = load('motorFailure_prob2_1_ksomef_hpo_none_norm3_1nn.mat');
results05 = load('motorFailure_prob2_1_ksomef_hpo_random_norm3_1nn.mat');
results06 = load('motorFailure_prob2_1_ksomgd_hpo_none_norm3_1nn.mat');
results07 = load('motorFailure_prob2_1_ksomgd_hpo_random_norm3_1nn.mat');
results08 = load('motorFailure_prob2_2_ksomef_hpo_none_norm_0_nn_1_ep_200.mat');
results09 = load('motorFailure_prob2_2_ksomef_hpo_random_norm3_1nn.mat');
results10 = load('motorFailure_prob2_2_ksomef_hpo_random_norm_3_nn_1_ep_50_Kt_1.mat');
results11 = load('motorFailure_prob2_4_ksomgd_hpo_random_norm3_1nn.mat');

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


















