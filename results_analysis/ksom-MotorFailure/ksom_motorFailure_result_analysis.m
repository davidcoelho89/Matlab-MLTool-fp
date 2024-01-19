%% RESULT ANALYSIS

% KSOM Algorithms and Stationary Data Sets
% Author: David Nascimento Coelho
% Last Update: 2024/01/12

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% MOTOR FAILURE (02 - balanced), KSOMEF, HPO RANDOM, ONE KERNEL

clear; clc;

% One result (to see others, just modify the number after Kt [1 - 8]

results = load('motorFailure_prob2_2_ksomef_hpo_random_norm_3_lbl_1_nn_1_Nep_200_Kt_2.mat');
% results = load('motorFailure_2_ksomef_hpo_random_norm_3_lbl_1_nn_1_Nep_50_Nprot_20_Kt_1.mat');
class_stats_ncomp(results.nstats_all,results.NAMES);

% All results - for lbl 1

results_01 = load('motorFailure_2_ksomef_hpo_random_norm_3_lbl_1_nn_1_Nep_50_Nprot_20_Kt_1.mat');
results_02 = load('motorFailure_2_ksomef_hpo_random_norm_3_lbl_1_nn_1_Nep_50_Nprot_20_Kt_2.mat');
results_03 = load('motorFailure_2_ksomef_hpo_random_norm_3_lbl_1_nn_1_Nep_50_Nprot_20_Kt_3.mat');
results_04 = load('motorFailure_2_ksomef_hpo_random_norm_3_lbl_1_nn_1_Nep_50_Nprot_20_Kt_4.mat');
results_05 = load('motorFailure_2_ksomef_hpo_random_norm_3_lbl_1_nn_1_Nep_50_Nprot_20_Kt_5.mat');
% results_06 = load('motorFailure_2_ksomef_hpo_random_norm_3_lbl_1_nn_1_Nep_50_Nprot_20_Kt_6.mat');
% results_07 = load('motorFailure_2_ksomef_hpo_random_norm_3_lbl_1_nn_1_Nep_50_Nprot_20_Kt_7.mat');
% results_08 = load('motorFailure_2_ksomef_hpo_random_norm_3_lbl_1_nn_1_Nep_50_Nprot_20_Kt_8.mat');

nstats_all = cell(5,1);
nstats_all{1,1} = results_01.ksomef_nstats_ts;
nstats_all{2,1} = results_02.ksomef_nstats_ts;
nstats_all{3,1} = results_03.ksomef_nstats_ts;
nstats_all{4,1} = results_04.ksomef_nstats_ts;
nstats_all{5,1} = results_05.ksomef_nstats_ts;
% nstats_all{6,1} = results_06.ksomef_nstats_ts;
% nstats_all{7,1} = results_07.ksomef_nstats_ts;
% nstats_all{8,1} = results_08.ksomef_nstats_ts;

NAMES = {'Linear','Gaussian',...    
         'Polynomial', 'Exponential',...
         'Cauchy'... ; 'Log', ...
         % 'Sigmoid', 'Kmod'};
         };

class_stats_ncomp(nstats_all,NAMES);

% All results - for lbl 2



% All Results - for lbl 3



%% MOTOR FAILURE (01 - unbalanced), KSOMEF, HPO RANDOM, VARIOUS KERNELS

% Load Results
results = load('motorFailure_1_ksomef_hpo_random_norm_3_lbl_1_nn_1_Kt_all.mat');

% Kernel Names
NAMES = {'Linear','Gaussian',... 
         'Polynomial', 'Exponential',...
         'Cauchy', 'Log',...
         'Sigmoid', 'Kmod'};

% Train Stats
class_stats_ncomp(results.variables.nstats_all_tr,NAMES);

% Test Stats
class_stats_ncomp(results.variables.nstats_all_ts,NAMES);


%% 


%% END


















