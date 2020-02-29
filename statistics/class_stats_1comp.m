function [nSTATS] = class_stats_1comp(STATS,NAMES)

% --- Provide comparison of 1 turn of Classifiers ---
%
%   [nSTATS] = class_stats_1comp(STATS,NAMES)
% 
%   Input:
%    	STATS = Cell containing statistics of various classifiers
%       NAMES = Cell containing names of various classifiers
%   Output:
%       nSTATS.

%% INITIALIZATIONS

% Get number of models and turns
[n_models,~] = size(NAMES);
[~,t] = size(STATS{1,1}.acc);

% Init Outputs
Mat_boxplot = zeros(t,n_models);

%% ALGORITHM

% Box Plot

for i = 1:n_models,
    Mat_boxplot(:,i) = STATS{i}.acc';
end

figure; boxplot(Mat_boxplot, 'label', NAMES);
set(gcf,'color',[1 1 1])        % Removes Gray Background
ylabel('Accuracy')
xlabel('Classifiers')
title('Training Results')
axis ([0 n_models+1 0 1.05])

hold on
media1 = mean(Mat_boxplot1);    % Mean Accuracy rate
plot(media1,'*k')
hold off

%% FILL OUTPUT STRUCTURE

nSTATS = STATS + NAMES;

%% END