%% RESULT ANALYSIS

% SPARK and Stationary Data Sets
% Author: David Nascimento Coelho
% Last Update: 2024/06/11

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% WALL FOLLOWING, SPARK, VARIOUS KERNELS

% Init

close;
clear;
clc;

% Choices for filename:

% str1 = 'wallFollow_1_spok_hold_2_norm_3_hpo_1_Dm';
str1 = 'wallFollow_1_spok_hold_2_norm_3_hpo_b_Dm';

dm = {'1','2'};
ss = {'1','2','3','4'};
kt = {'lin','gau','pol','exp','cau','log','sig','kmod'};
knn = {'1','2'};

% Get number of realizations

i = 1; j = 1; k = 1; l = 1;
filename = strcat(str1,dm{i},...
                  '_Ss',ss{j},...
                  '_Us0_Ps0_',...
                  kt{k},'_',...
                  'nn',knn{l}, ...
                  '.mat');
variables = load(filename);
Nr = variables.OPT.Nr;

clear variables;

% Init variables

lines = length(ss) * length(dm) * length(knn);

mat_acc_mean = zeros(lines,length(kt));
mat_acc_median = zeros(lines,length(kt));
mat_acc_best = zeros(lines,length(kt));
mat_acc_std = zeros(lines,length(kt));
mat_acc_boxplot = zeros(Nr,length(kt));

mat_fsc_best = zeros(lines,length(kt));
mat_mcc_best = zeros(lines,length(kt));

mat_nprot_best = zeros(lines,length(kt));
mat_nprot_mean = zeros(lines,length(kt));

mat_K_best = zeros(lines,length(kt));
mat_K_mean = zeros(lines,length(kt));

% Obs: 13 are the number of optimized HPs (considering all kernels)
mat_hp_best = zeros(lines,13); 

mat_v1_v2_best = zeros(lines,2*length(kt));

% plot number of times a certain value of HP was chosen.

cell_best_hps = cell(lines,length(kt));

line = 0;
for j = 1:length(ss)
    for i = 1:length(dm)
        for l = 1:length(knn)
            line = line + 1;
            for k = 1:length(kt)
                
                best_hps = struct();
                if(k == 1)
                    best_hps.theta = zeros(1,Nr);
                elseif(k == 2)
                    best_hps.sigma = zeros(1,Nr);
                elseif(k == 3)
                    best_hps.alpha = zeros(1,Nr);
                    best_hps.theta = zeros(1,Nr);
                    best_hps.gamma = zeros(1,Nr);
                elseif(k == 4)
                    best_hps.sigma = zeros(1,Nr);
                elseif(k == 5)
                    best_hps.sigma = zeros(1,Nr);
                elseif(k == 6)
                    best_hps.sigma = zeros(1,Nr);
                    best_hps.gamma = zeros(1,Nr);
                elseif(k == 7)
                    best_hps.alpha = zeros(1,Nr);
                    best_hps.theta = zeros(1,Nr);
                elseif(k == 8)
                    best_hps.sigma = zeros(1,Nr);
                    best_hps.gamma = zeros(1,Nr);
                end
                best_hps.v1 = zeros(1,Nr);
                best_hps.v2 = zeros(1,Nr);
                cell_best_hps{line,k} = best_hps;
                
            end
        end
    end
end

% Get Values

line = 0;
for j = 1:length(ss)
    for i = 1:length(dm)
        for l = 1:length(knn)
            
            line = line + 1;
            
            for k = 1:length(kt)
                
                % Generate filenames
                
                filename = strcat(str1,dm{i}, ...
                                  '_Ss',ss{j}, ...
                                  '_Us0_Ps0_', ...
                                  kt{k},'_', ...
                                  'nn',knn{l}, ...
                                  '.mat');
                disp(filename);

                % Get variables
               
                variables = load(filename);
                
                if(isfield(variables,'nSTATS_ts'))
                    nstats_ts = variables.nSTATS_ts;
                elseif(isfield(variables,'nstats_ts'))
                    nstats_ts = variables.nstats_ts;
                end
                
                if(isfield(variables,'STATS_ts_acc'))
                    stats_ts_acc = variables.STATS_ts_acc;
                elseif(isfield(variables,'stats_ts_acc'))
                    stats_ts_acc = variables.stats_ts_acc;
                end
                
                if(isfield(variables,'PAR_acc'))
                    par_acc = variables.PAR_acc;
                elseif(isfield(variables,'par_acc'))
                    par_acc = variables.par_acc;
                end
                                
                best_acc_index = nstats_ts.acc_max_i;

                % Update acc matrices
                
                mat_acc_best(line,k) = stats_ts_acc{best_acc_index,1}.acc;
                mat_acc_mean(line,k) = nstats_ts.acc_mean;
                mat_acc_median(line,k) = nstats_ts.acc_median;
                mat_acc_std(line,k) = nstats_ts.acc_std;
                mat_acc_boxplot(:,k) = nstats_ts.acc';
                
                % Update mcc and fsc
                
                mat_fsc_best(line,k) = stats_ts_acc{best_acc_index,1}.fsc_macro;
                mat_mcc_best(line,k) = stats_ts_acc{best_acc_index,1}.mcc_multiclass;
                
                % Update Hyperparameters
                
                mat_nprot_best(line,k) = size(par_acc{best_acc_index,1}.Cx,2);
                mat_K_best(line,k) = par_acc{best_acc_index,1}.K;
                
                mat_v1_v2_best(line,2*k-1) = par_acc{best_acc_index,1}.v1;
                mat_v1_v2_best(line,2*k) = par_acc{best_acc_index,1}.v2;

                if(k == 1)
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

                for m = 1:Nr
                    mat_nprot_mean(line,k) = mat_nprot_mean(line,k) + ...
                                  size(par_acc{m,1}.Cx,2);
                    mat_K_mean(line,k) = mat_K_mean(line,k) + ...
                                  par_acc{m,1}.K;
                              
                    if(k == 1)
                        cell_best_hps{line,k}.theta(1,m) = par_acc{m,1}.theta;
                    elseif(k == 2)
                        cell_best_hps{line,k}.sigma(1,m) = par_acc{m,1}.sigma;
                    elseif(k == 3)
                        cell_best_hps{line,k}.alpha(1,m) = par_acc{m,1}.alpha;
                        cell_best_hps{line,k}.theta(1,m) = par_acc{m,1}.theta;
                        cell_best_hps{line,k}.gamma(1,m) = par_acc{m,1}.gamma;
                    elseif(k == 4)
                        cell_best_hps{line,k}.sigma(1,m) = par_acc{m,1}.sigma;
                    elseif(k == 5)
                        cell_best_hps{line,k}.sigma(1,m) = par_acc{m,1}.sigma;
                    elseif(k == 6)
                        cell_best_hps{line,k}.gamma(1,m) = par_acc{m,1}.gamma;
                        cell_best_hps{line,k}.sigma(1,m) = par_acc{m,1}.sigma;
                    elseif(k == 7)
                        cell_best_hps{line,k}.alpha(1,m) = par_acc{m,1}.alpha;
                        cell_best_hps{line,k}.theta(1,m) = par_acc{m,1}.theta;
                    elseif(k == 8)
                        cell_best_hps{line,k}.gamma(1,m) = par_acc{m,1}.gamma;
                        cell_best_hps{line,k}.sigma(1,m) = par_acc{m,1}.sigma;
                    end
                    cell_best_hps{line,k}.v1(1,m) = par_acc{m,1}.v1;
                    cell_best_hps{line,k}.v2(1,m) = par_acc{m,1}.v2;
                
                end
            
                % Clear variables;
                clear variables;
                
            end
            
%             % Generate Accuracy boxplot (1 line)
%             figure; boxplot(mat_boxplot_acc, 'label', kt);
%             set(gcf,'color',[1 1 1])        % Removes Gray Background
%             ylabel('Accuracy')
%             xlabel('Kernels')
%             title_str = strcat('Accuracy-','Ss',ss{j},'Dm',dm{i},'nn',knn{l});
%             title(title_str)
%             axis ([0 length(kt)+1 -0.05 1.05])
% 
%             hold on
%             plot(mean(mat_boxplot_acc),'*k')
%             hold off
% 
%             % Generate #Prot boxplot (1 line)
%             figure; boxplot(mat_boxplot_nprot, 'label', kt);
%             set(gcf,'color',[1 1 1])        % Removes Gray Background
%             ylabel('Accuracy')
%             xlabel('Kernels')
%             title_str = strcat('#Prot-','Ss',ss{j},'Dm',dm{i},'nn',knn{l});
%             title(title_str)
%             min_nprot = min(min(mat_boxplot_nprot));
%             max_nprot = max(max(mat_boxplot_nprot));
%             axis ([0 length(kt)+1 min_nprot-1 max_nprot+1])
                
        end
    end
end
mat_nprot_mean = mat_nprot_mean/Nr;
mat_K_mean = mat_K_mean/Nr;

% disp(mat_nprot_mean);
% disp(mat_K_mean);

%% END