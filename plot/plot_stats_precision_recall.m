function [] = plot_stats_precision_recall(STATS)

% --- Plot one Precision-Recall Curve for each class ---
%
%	[] = plot_stats_precision_recall(STATS)
%
%   Input:
%       STATS.
%           roc_prec = Precision                                [Nc x len]
%           roc_rec = Recall (True Positive Rate, Sensitivity)	[Nc x len] 
%   Output:
%       "void" (print a graphic at screen)

%% INITIALIZATIONS

% Get number of classes

[Nc,~] = size(STATS.roc_tpr);

% Get curves

roc_prec = STATS.roc_prec;
roc_rec = STATS.roc_rec;

%% ALGORITHM

for c = 1:Nc
    figure;
    hold on
    plot(roc_rec(c,:),roc_prec(c,:),'r.-');
    plot([1,1,0],[0,1,1],'k-');
    axis([-0.1 1.1 -0.1 1.1])
    s1 = 'Precision-Recall Curve: Class ';   s2 = int2str(c);
    title(strcat(s1,s2))
    xlabel('Precision')
    ylabel('Recall')
    hold off
end

%% FILL OUTPUT STRUCTURE

% Don't have output structure

%% END