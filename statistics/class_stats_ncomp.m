function [] = class_stats_ncomp(STATS,NAMES)

% --- Provide comparison of n turns of Classifiers ---
%
%   [] = class_stats_ncomp(STATS,NAMES)
% 
%   Input:
%    	STATS = Cell containing statistics of various classifiers
%       NAMES = Cell containing names of various classifiers
%   Output:
%       "void" (print a graphic at screen)

%% INITIALIZATIONS

% Get number of models and turns
[n_models,~] = size(STATS);
[~,n_turns] = size(STATS{1,1}.acc);

% If it dont have names
if (nargin == 1)
    NAMES = cell(1,n_models);
    for i = 1:n_models
        NAMES{i} = strcat('class ',int2str(i));
    end
end

% Init Outputs
Mat_boxplot_acc = zeros(n_turns,n_models);
Mat_boxplot_err = zeros(n_turns,n_models);
Mat_boxplot_fsc = zeros(n_turns,n_models);
Mat_boxplot_mcc = zeros(n_turns,n_models);

%% ALGORITHM

% Box Plot - Accuracy

for i = 1:n_models
    Mat_boxplot_acc(:,i) = STATS{i}.acc';
end

figure; boxplot(Mat_boxplot_acc, 'label', NAMES);
set(gcf,'color',[1 1 1])        % Removes Gray Background
ylabel('Accuracy')
xlabel('Classifiers')
title('Classification Results')
axis ([0 n_models+1 -0.05 1.05])

hold on
plot(mean(Mat_boxplot_acc),'*k')
hold off

% Box Plot - Error

for i = 1:n_models
    Mat_boxplot_err(:,i) = STATS{i}.err';
end

figure; boxplot(Mat_boxplot_err, 'label', NAMES);
set(gcf,'color',[1 1 1])        % Removes Gray Background
ylabel('Error')
xlabel('Classifiers')
title('Classification Results')
axis ([0 n_models+1 -0.05 1.05])

hold on
plot(mean(Mat_boxplot_err),'*k')
hold off

% Box Plot - F1-score

for i = 1:n_models
    Mat_boxplot_fsc(:,i) = STATS{i}.fsc';
end

figure; boxplot(Mat_boxplot_fsc, 'label', NAMES);
set(gcf,'color',[1 1 1])        % Removes Gray Background
ylabel('F1-Score (macro-averaged)')
xlabel('Classifiers')
title('Classification Results')
axis ([0 n_models+1 -0.05 1.05])

hold on
plot(mean(Mat_boxplot_fsc),'*k')
hold off

% Box Plot - Matthews Correlation Coefficient

for i = 1:n_models
    Mat_boxplot_mcc(:,i) = STATS{i}.mcc';
end

figure; boxplot(Mat_boxplot_mcc, 'label', NAMES);
set(gcf,'color',[1 1 1])        % Removes Gray Background
ylabel('MCC (multiclass)')
xlabel('Classifiers')
title('Classification Results')
axis ([0 n_models+1 -0.05 1.05])

hold on
plot(mean(Mat_boxplot_mcc),'*k')
hold off

%% FILL OUTPUT STRUCTURE

% Don't have output structure

%% END