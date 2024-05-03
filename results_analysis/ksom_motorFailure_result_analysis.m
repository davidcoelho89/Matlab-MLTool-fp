%% RESULT ANALYSIS

% KSOM Algorithms and Stationary Data Sets
% Author: David Nascimento Coelho
% Last Update: 2024/01/12

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% MOTOR FAILURE (02 - balanced), KSOM, HOLD 2, HPO RANDOM, VARIOUS KERNELS

% Init

close;
clear;
clc;

% Choices for filename

str1 = 'motorFailure_2_';
ksom = {'ksomef','ksomgd'};
str2 = '_hold_1_hpo_1_norm_3_lbl_';
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
mat_acc_boxplot = zeros(Nr,length(kt));

mat_fsc_best = zeros(lines,length(kt));
mat_mcc_best = zeros(lines,length(kt));

mat_hp_best = zeros(lines,13); % Obs: 13 are the number of optimized HPs (considering all kernels)

% Get values

line = 0;
for i = 2:length(ksom)
    for j = 1:length(lbl)

        line = line + 1;
        
        for k = 1:length(kt)
            
            % generate filenames
            filename = strcat(str1,ksom{i},str2,lbl{j},str3,kt{k});
            disp(filename);
            
            % Get variables
            variables = load(filename);
            best_acc_index = variables.nstats_all{2,1}.acc_max_i;
            
            % Update acc matrices            
            mat_acc_best(line,k) = variables.nstats_all{2,1}.acc(best_acc_index);
            mat_acc_mean(line,k) = variables.nstats_all{2,1}.acc_mean;
            mat_acc_median(line,k) = variables.nstats_all{2,1}.acc_median;
            mat_acc_boxplot(:,k) = variables.nstats_all{2,1}.acc';
            
            % Update mcc and fcc
            mat_fsc_best(line,k) = variables.nstats_all{2,1}.mcc(best_acc_index);
            mat_mcc_best(line,k) = variables.nstats_all{2,1}.fsc(best_acc_index);
            
            % Update Hyperparameters
            if (k == 1)
                mat_hp_best(line,1) = variables.ksomgd_par_acc{best_acc_index,1}.theta;
            elseif(k == 2)
                mat_hp_best(line,2) = variables.ksomgd_par_acc{best_acc_index,1}.sigma;
            elseif(k == 3)
                mat_hp_best(line,3) = variables.ksomgd_par_acc{best_acc_index,1}.alpha;
                mat_hp_best(line,4) = variables.ksomgd_par_acc{best_acc_index,1}.theta;
                mat_hp_best(line,5) = variables.ksomgd_par_acc{best_acc_index,1}.gamma;
            elseif(k == 4)
                mat_hp_best(line,6) = variables.ksomgd_par_acc{best_acc_index,1}.sigma;
            elseif(k == 5)
                mat_hp_best(line,7) = variables.ksomgd_par_acc{best_acc_index,1}.sigma;
            elseif(k == 6)
                mat_hp_best(line,8) = variables.ksomgd_par_acc{best_acc_index,1}.gamma;
                mat_hp_best(line,9) = variables.ksomgd_par_acc{best_acc_index,1}.sigma;
            elseif(k == 7)
                mat_hp_best(line,10) = variables.ksomgd_par_acc{best_acc_index,1}.alpha;
                mat_hp_best(line,11) = variables.ksomgd_par_acc{best_acc_index,1}.theta;
            elseif(k == 8)
                mat_hp_best(line,12) = variables.ksomgd_par_acc{best_acc_index,1}.gamma;
                mat_hp_best(line,13) = variables.ksomgd_par_acc{best_acc_index,1}.sigma;
            end
            
            % Clear variables;
            clear variables;
            
        end
    end
end

%% END


















