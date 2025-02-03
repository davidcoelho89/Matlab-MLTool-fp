%% RESULT ANALYSIS

% Analysis of results from ksom model and CervicalCancer dataset
% Author: David Nascimento Coelho
% Last Update: 2024/05/27

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% CERVICAL CANCER (02 - binary), KSOM, HOLD 1, HPO RANDOM or BEST, VARIOUS KERNELS

% Init

close;
clear;
clc;

% Choices for filename

str1 = 'cervicalCancer_2_';
ksom = {'ksomef','ksomgd'};
% str2 = '_hold_1_norm_3_hpo_1_lbl_';
str2 = '_hold_1_norm_3_hpo_b_lbl_';
lbl = {'1','2','3'};
str3 = '_nn_1_Nep_50_Nprot_30_Kt_';
kt = {'1','2','3','4','5','6','7','8'};

% Get number of realizations
i = 1; j = 1; k = 1;
filename = strcat(str1,ksom{i},str2,lbl{j},str3,kt{k});
variables = load(filename);
Nr = variables.OPT.Nr;
clear variables;

% Init variables

lines = length(ksom) * length(lbl);

mat_acc_mean = zeros(lines,length(kt));
mat_acc_median = zeros(lines,length(kt));
mat_acc_best = zeros(lines,length(kt));
mat_acc_std = zeros(lines,length(kt));
mat_acc_boxplot = zeros(Nr,length(kt));

mat_fsc_best = zeros(lines,length(kt));
mat_mcc_best = zeros(lines,length(kt));

mat_hp_best = zeros(lines,13); % Obs: 13 are the number of optimized HPs (considering all kernels)

% Get values

line = 0;
for i = 1:length(ksom)
    for j = 1:length(lbl)

        line = line + 1;
        
        for k = 1:length(kt)
            
            % Get variables from file
            filename = strcat(str1,ksom{i},str2,lbl{j},str3,kt{k});
            variables = load(filename);
            disp(filename);
            
            % Update acc matrices (from test)
            best_acc_index = variables.nstats_all{2,1}.acc_max_i;
            mat_acc_best(line,k) = variables.nstats_all{2,1}.acc(best_acc_index);
            mat_acc_mean(line,k) = variables.nstats_all{2,1}.acc_mean;
            mat_acc_median(line,k) = variables.nstats_all{2,1}.acc_median;
            mat_acc_std(line,k) = variables.nstats_all{2,1}.acc_std;
            mat_acc_boxplot(:,k) = variables.nstats_all{2,1}.acc';
            
            % Update mcc and fcc (from test)
            mat_fsc_best(line,k) = variables.nstats_all{2,1}.mcc(best_acc_index);
            mat_mcc_best(line,k) = variables.nstats_all{2,1}.fsc(best_acc_index);
            
            % Update Hyperparameters
            if(isfield(variables,'ksomgd_par_acc'))
                par_acc = variables.ksomgd_par_acc;
            elseif(isfield(variables,'ksomef_par_acc'))
                par_acc = variables.ksomef_par_acc;
            elseif(isfield(variables,'par_acc'))
                par_acc = variables.par_acc;
            end            
            
            if (k == 1)
                mat_hp_best(line,1) = par_acc{best_acc_index,1}.theta;
            elseif(k == 2)
                mat_hp_best(line,2) = par_acc{best_acc_index,1}.sigma;
            elseif(k == 3)
                mat_hp_best(line,3) = par_acc{best_acc_index,1}.alpha;
                mat_hp_best(line,4) = par_acc{best_acc_index,1}.theta;
                mat_hp_best(line,5) = par_acc{best_acc_index,1}.gamma;
            elseif(k == 4)
                mat_hp_best(line,6) = par_acc{best_acc_index,1}.sigma;
            elseif(k == 5)
                mat_hp_best(line,7) = par_acc{best_acc_index,1}.sigma;
            elseif(k == 6)
                mat_hp_best(line,8) = par_acc{best_acc_index,1}.gamma;
                mat_hp_best(line,9) = par_acc{best_acc_index,1}.sigma;
            elseif(k == 7)
                mat_hp_best(line,10) = par_acc{best_acc_index,1}.alpha;
                mat_hp_best(line,11) = par_acc{best_acc_index,1}.theta;
            elseif(k == 8)
                mat_hp_best(line,12) = par_acc{best_acc_index,1}.gamma;
                mat_hp_best(line,13) = par_acc{best_acc_index,1}.sigma;
            end
            
            % Clear variables;
            clear variables;
           
        end
    end
end

clc;

%% END