function [] = plot_stats_roc_curve(STATS)

% --- Plot one ROC Curve (1 - spec x sens) for each class ---
%
%	[] = plot_stats_roc_curve(STATS)
%
%   Input:
%       STATS.
%           roc_tpr = true positive rate (sensitivity)      [Nc x len] 
%           roc_fpr = false positive rate (1 - specificity) [Nc x len]
%   Output:
%       "void" (print a graphic at screen)

%% INITIALIZATIONS

% Get number of classes

[Nc,~] = size(STATS.roc_tpr);

% Get curves

roc_tpr = STATS.roc_tpr;
roc_fpr = STATS.roc_fpr;

%% ALGORITHM

for c = 1:Nc
    figure;
    hold on
    plot(roc_fpr(c,:),roc_tpr(c,:),'r.-');
    plot([0,0,1],[0,1,1],'k-');
    axis([-0.1 1.1 -0.1 1.1])
    s1 = 'ROC Curve: Class ';   s2 = int2str(c);
    title(strcat(s1,s2))
    xlabel('1 - Specificity')
    ylabel('Sensitity')
    hold off
end

%% FILL OUTPUT STRUCTURE

% Don't have output structure

%% END