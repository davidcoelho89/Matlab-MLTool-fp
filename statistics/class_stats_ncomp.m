function [] = class_stats_ncomp(STATS,NAMES)

% --- Provide comparison of n turns of Classifiers ---
%
%   [nSTATS] = class_stats_ncomp(STATS,NAMES)
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
Mat_boxplot1 = zeros(n_turns,n_models);
Mat_boxplot2 = zeros(n_turns,n_models);
% Mat_boxplot3 = zeros(n_turns,n_models);

%% ALGORITHM

% Box Plot - Accuracy

for i = 1:n_models
    Mat_boxplot1(:,i) = STATS{i}.acc';
end

figure; boxplot(Mat_boxplot1, 'label', NAMES);
set(gcf,'color',[1 1 1])        % Removes Gray Background
ylabel('Accuracy')
xlabel('Classifiers')
title('Classification Results')
axis ([0 n_models+1 -0.05 1.05])

hold on
media1 = mean(Mat_boxplot1);    % Mean Accuracy rate
plot(media1,'*k')
hold off

% Box Plot - Error

for i = 1:n_models
    Mat_boxplot2(:,i) = STATS{i}.err';
end

figure; boxplot(Mat_boxplot2, 'label', NAMES);
set(gcf,'color',[1 1 1])        % Removes Gray Background
ylabel('Error')
xlabel('Classifiers')
title('Classification Results')
axis ([0 n_models+1 -0.05 1.05])

hold on
media1 = mean(Mat_boxplot2);    % Mean Accuracy rate
plot(media1,'*k')
hold off

% Box Plot - Mcc (for class 1)

% for i = 1:n_models,
%     for j = 1:n_turns,
%         Mat_boxplot3(j,i) = STATS{i}.mcc{j}(1);
%     end
% end
% 
% figure; boxplot(Mat_boxplot3, 'label', NAMES);
% set(gcf,'color',[1 1 1])        % Removes Gray Background
% ylabel('Mcc')
% xlabel('Classifiers')
% title('Matthews Correlation Coefficient')
% axis ([0 n_models+1 -0.05 1.05])
% 
% hold on
% media1 = mean(Mat_boxplot2);    % Mean Accuracy rate
% plot(media1,'*k')
% hold off

% Box Plot - Auc (for class 1)

%% FILL OUTPUT STRUCTURE

% Don't have output structure

%% END