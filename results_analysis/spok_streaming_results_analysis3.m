%% RESULT ANALYSIS

% SPOK and Streaming Data Sets
% Author: David Nascimento Coelho
% Last Update: 2024/10/26

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% SPOK, 1 DATASET, VARIOUS KERNELS

% Init

close;
clear;
clc;

% Choices for filename:
% chess_1_spok_hold_ttt_norm_0_hpo_1_Dm_2_Ss_1_Us_1_Ps_2_cau_nn1
% coverType_1_spok_hold_ttt_norm_0_hpo_1_Dm_2_Ss_1_Us_1_Ps_2_cau_nn1
% electricity_1_spok_hold_ttt_norm_0_hpo_1_Dm_2_Ss_1_Us_1_Ps_2_cau_nn1
% outdoor_1_spok_hold_ttt_norm_0_hpo_1_Dm_2_Ss_1_Us_1_Ps_2_cau_nn1
% poker_1_spok_hold_ttt_norm_0_hpo_1_Dm_2_Ss_1_Us_1_Ps_2_cau_nn1
% rbfInt_1_spok_hold_ttt_norm_0_hpo_1_Dm_2_Ss_1_Us_1_Ps_2_cau_nn1
% rialto_1_spok_hold_ttt_norm_0_hpo_1_Dm_2_Ss_1_Us_1_Ps_2_cau_nn1
% squaresMov_1_spok_hold_ttt_norm_0_hpo_1_Dm_2_Ss_1_Us_1_Ps_2_cau_nn1
% weather_1_spok_hold_ttt_norm_0_hpo_1_Dm_2_Ss_1_Us_1_Ps_2_cau_nn1

str1 = 'chess'; % 'coverType' 'electricity' 'outdoor' 'poker' 
                % 'rbfInt' 'rialto' 'squaresMov' 'weather'
str2 = '_1_spok_hold_ttt_norm_0_hpo_1_Dm_2_Ss_';
ss = {'1','2','3','4'};
kt = {'lin','gau','pol','exp','cau','log','sig','kmod'};
knn = {'1','2'};

% Get info from first file

i = 1; j = 1; k = 1;
filename = strcat(str1,str2,ss{i},...
                  '_Us_1_Ps_2_',...
                  kt{k},'_',...
                  'nn',knn{j},...
                  '.mat');

variables = load(filename);

% Init Variables

lines = length(ss) * length(knn);
nkernels = length(kt);

mat_acc_final = zeros(lines,nkernels);
mat_acc_mean = zeros(lines,nkernels);
mat_acc_best = zeros(lines,nkernels);
mat_acc_std = zeros(lines,nkernels);

mat_nprot_final = zeros(lines,nkernels);
mat_nprot_mean = zeros(lines,nkernels);
mat_nprot_std = zeros(lines,nkernels);

mat_hp_kernel_best = zeros(lines,13); % Obs: 13 are the number of optim
                                      % HPs (considering all kernels)

mat_hp_v1_v2_best = zeros(lines,2*nkernels);

mat_k = zeros(lines,nkernels);

% Get Values

line = 0;
for i = 1:length(ss)
    for j = 1:length(knn)
        
        line = line + 1;
        
        for k = 1:nkernels
            
            % Generate filename
            filename = strcat(str1,str2,ss{i},...
                              '_Us_1_Ps_2_',...
                              kt{k},'_',...
                              'nn',knn{j},...
                              '.mat');
            disp(filename);
            
            % Get variables
            
            variables = load(filename);
            
            acc_vector = variables.accuracy_vector;
            
            mat_acc_final(line,k) = acc_vector(end);
            mat_acc_mean(line,k) = mean(acc_vector);
            mat_acc_best(line,k) = max(acc_vector(1000:end));
            mat_acc_std(line,k) = std(acc_vector);
            
            prot_vector = variables.prot_per_class(end,:);
            
            mat_nprot_final(line,k) = prot_vector(end);
            mat_nprot_mean(line,k) = mean(prot_vector);
            mat_nprot_std(line,k) = std(prot_vector);
            
            par = variables.PAR;
            
            mat_k(line,k) = par.K;
            
            mat_hp_v1_v2_best(line,2*k-1) = par.v1;
            mat_hp_v1_v2_best(line,2*k) = par.v2;
            
            if (k == 1)
                mat_hp_kernel_best(line,1) = par.theta;
            elseif (k == 2)
                mat_hp_kernel_best(line,2) = par.sigma;
            elseif (k == 3)
                mat_hp_kernel_best(line,3) = par.alpha;
                mat_hp_kernel_best(line,4) = par.theta;
                mat_hp_kernel_best(line,5) = par.gamma;
            elseif (k == 4)
                mat_hp_kernel_best(line,6) = par.sigma;
            elseif (k == 5)
                mat_hp_kernel_best(line,7) = par.sigma;
            elseif (k == 6)
                mat_hp_kernel_best(line,8) = par.gamma;
                mat_hp_kernel_best(line,9) = par.sigma;
            elseif (k == 7)
                mat_hp_kernel_best(line,10) = par.alpha;
                mat_hp_kernel_best(line,11) = par.theta;
            elseif (k == 8)
                mat_hp_kernel_best(line,12) = par.gamma;
                mat_hp_kernel_best(line,13) = par.sigma;
            end
            
            % Clear variables;
            clear variables;
            clear acc_vector;
            clear prot_vector;
        end
        
    end
end


%% END





















