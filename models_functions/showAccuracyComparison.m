function [] = showAccuracyComparison(statsComp,names)

% --- Provide accuracy comparison of n turns of Classifiers ---
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
[~,n_stats] = size(statsComp);
[~,n_turns] = size(statsComp{1,1}.acc_vect);

% If it dont have names
if (nargin == 1)
    names = cell(1,n_stats);
    for i = 1:n_stats
        names{i} = strcat('class ',int2str(i));
    end
end

% Init Outputs
Mat_boxplot1 = zeros(n_turns,n_stats);

%% ALGORITHM

% Box Plot - Accuracy

for i = 1:n_stats
    Mat_boxplot1(:,i) = statsComp{i}.acc_vect';
end

figure; boxplot(Mat_boxplot1, 'label', names);
set(gcf,'color',[1 1 1])        % Removes Gray Background
ylabel('Accuracy')
xlabel('Classifiers')
title('Classification Results')
axis ([0 n_stats+1 -0.05 1.05])

hold on
media1 = mean(Mat_boxplot1);    % Mean Accuracy rate
plot(media1,'*k')
hold off

%% FILL OUTPUT STRUCTURE

% Don't have output structure

%% END