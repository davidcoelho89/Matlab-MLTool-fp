%% RESULT ANALYSIS

% SPARK and Stationary Data Sets
% Author: David Nascimento Coelho
% Last Update: 2024/01/30

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% MOTOR FAILURE (02 - balanced), SPARK, HPO RANDOM, VARIOUS KERNELS

% Init

close;
clear;
clc;

% Choices for filename

str1 = 'motorFailure_isk2nn_hpo1_norm3_Dm';

dm = {'1','2'};
ss = {'1','2','3','4'};
kt = {'lin','gau','pol','exp','cau','log','sig','kmod'};
knn = {'1','2'};

% Get number of realizations

i = 1; j = 1; k = 1; l = 1;
filename = strcat(str1,dm{i},'_Ss',ss{j},'_Us0_Ps0_',kt{k},'_',knn{l},'nn');
variables = load(filename);
Nr = variables.OPT.Nr;

% Init variables

line = 0;
lines = length(ss) * length(dm) * length(knn);

mat_acc_mean = zeros(lines,length(kt));
mat_acc_median = zeros(lines,length(kt));
mat_acc_best = zeros(lines,length(kt));
mat_acc_boxplot = zeros(Nr,length(kt));
mat_nprot_best = zeros(lines,length(kt));
mat_nprot_mean = zeros(lines,length(kt));
mat_K_best = zeros(lines,length(kt));
mat_K_mean = zeros(lines,length(kt));

% Get Values


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
                                  knn{l},'nn');
                disp(filename);
                % Get variables
                variables = load(filename);
                best_acc_index = variables.nSTATS_ts.acc_max_i;
                % Update acc matrices
                mat_acc_best(line,k) = variables.STATS_ts_acc{best_acc_index,1}.acc;
                mat_acc_mean(line,k) = variables.nSTATS_ts.acc_mean;
                mat_acc_median(line,k) = variables.nSTATS_ts.acc_median;
                mat_acc_boxplot(:,k) = variables.nSTATS_ts.acc';
                % Update Nprot
                mat_nprot_best(line,k) = size(variables.PAR_acc{best_acc_index,1}.Cx,2);
                mat_K_best(line,k) = variables.PAR_acc{best_acc_index,1}.K;
                for m = 1:Nr
                    mat_nprot_mean(line,k) = mat_nprot_mean(line,k) + ...
                                  size(variables.PAR_acc{m,1}.Cx,2);
                    mat_K_mean(line,k) = mat_K_mean(line,k) + ...
                                  variables.PAR_acc{m,1}.K;
                end
                % Clear variables;
                clear variables;

            end

            % Generate Accuracy boxplot (1 line)
%             figure; boxplot(mat_acc_boxplot, 'label', kt);
%             set(gcf,'color',[1 1 1])        % Removes Gray Background
%             ylabel('Accuracy')
%             xlabel('Kernels')
%             title_str = strcat('Accuracy-','Ss',ss{j},'Dm',dm{i},'nn',knn{l});
%             title(title_str)
%             axis ([0 length(kt)+1 -0.05 1.05])
% 
%             hold on
%             plot(mean(mat_acc_boxplot),'*k')
%             hold off

        end
    end
end
mat_nprot_mean = mat_nprot_mean/Nr;
mat_K_mean = mat_K_mean/Nr;

%% MOTOR FAILURE (01 - unbalanced), SPARK, HPO RANDOM, LOG KERNEL

% Init

% close;
% clear;
% clc;


%% END