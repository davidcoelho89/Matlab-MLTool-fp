% %% RESULT ANALYSIS

% SPARK and Stationary Data Sets
% Author: David Nascimento Coelho
% Last Update: 2024/01/12

close;          % Close all windows
clear;          % Clear all variables
clc;            % Clear command window

format long e;  % Output data style (float)

%% MOTOR FAILURE (02 - balanced), SPARK, HPO RANDOM, VARIOUS KERNELS

% Init
close;
clear;
clc;

% Choices
dm = {'1','2'};
ss = {'1','2','3','4'};
kt = {'lin','gau','pol','exp','cau','log','sig','kmod'};
knn = {'1','2'};

% Init variables
line = 0;
lines = length(ss) * length(dm) * length(knn);
mat_mean_values = zeros(lines,length(kt));
mat_median_values = zeros(lines,length(kt));
mat_nprot = zeros(lines,length(kt));
mat_boxplot_acc = zeros(10,length(kt));

% i = 1; j = 1; k = 1; l = 1;
% line = 1;

% Get Values

str1 = 'motorFailure_isk2nn_hpo1_norm3_Dm';
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
                % Update matrices
                mat_mean_values(line,k) = variables.nSTATS_ts.acc_mean;
                mat_median_values(line,k) = variables.nSTATS_ts.acc_median;
                mat_boxplot_acc(:,k) = variables.nSTATS_ts.acc';
                % Clear variables;
                clear variables;

            end

            % Generate boxplot (1 line)
            figure; boxplot(mat_boxplot_acc, 'label', kt);
            set(gcf,'color',[1 1 1])        % Removes Gray Background
            ylabel('Accuracy')
            xlabel('Kernels')
            title_str = strcat('Ss',ss{j},'Dm',dm{i},'nn',knn{l});
            title(title_str)
            axis ([0 length(kt)+1 -0.05 1.05])

            hold on
            plot(mean(mat_boxplot_acc),'*k')
            hold off

        end
    end
end

%% MOTOR FAILURE (02 - balanced), SPARK, HPO RANDOM, LOG KERNEL

% Init
close;
clear;
clc;



%% END