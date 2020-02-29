function [nSTATS] = cluster_stats_nturns(STATS_acc)

% --- Provide Statistics of n turn of Clustering ---
%
%   [nSTATS] = cluster_stats_nturns(STATS_acc)
% 
%   Input:
%    	STATS_acc = Cell containing statistics of n turns of clustering
%   Output:
%       nSTATS.
%			ssqe = vector with ssqe for each turn               [1 x t]
%           msqe = vector with msqe for each turn               [1 x t]
%           aic = vector with aic index for each turn           [1 x t]
%           bic = vector with bic index for each turn           [1 x t]
%           ch = vector with ch index for each turn             [1 x t]
%           db = vector with db index for each turn             [1 x t]
%           dunn = vector with dunn index for each turn         [1 x t]
%           fpe = vector with fpe index for each turn           [1 x t]
%           mdl = vector with mdl index for each turn           [1 x t]
%           sil = vector with silhouette index for each turn 	[1 x t]

%% INITIALIZATIONS

% Get number of turns
[t,~] = size(STATS_acc);

% Init outputs
ssqe = zeros(1,t);
msqe = zeros(1,t);
aic = zeros(1,t);
bic = zeros(1,t);
ch = zeros(1,t);
db = zeros(1,t);
dunn = zeros(1,t);
fpe = zeros(1,t);
mdl = zeros(1,t);
sil = zeros(1,t);

ssqe_min = STATS_acc{1}.ssqe;
ssqe_max = STATS_acc{1}.ssqe;

%% ALGORITHM

for i = 1:t,
    % Get statistics from 1 turn
    STATS = STATS_acc{i};
    % ssqe vector
    ssqe(i) = STATS.ssqe;
    % msqe vector
    msqe(i) = STATS.msqe;
    % min ssqe
    if (STATS.ssqe <= ssqe_min),
        ssqe_min_index = i;
        ssqe_min = STATS.ssqe;
    end
    % max ssqe
    if (STATS.ssqe >= ssqe_max),
        ssqe_max_index = i;
        ssqe_max = STATS.ssqe;
    end
    % Cluster Validation indexes
    aic(i) = STATS.aic;
    bic(i) = STATS.bic;
    ch(i) = STATS.ch;
    db(i) = STATS.db;
    dunn(i) = STATS.dunn;
    fpe(i) = STATS.fpe;
    mdl(i) = STATS.mdl;
    sil(i) = STATS.sil;
end

%% FILL OUTPUT STRUCTURE

nSTATS.ssqe = ssqe;
nSTATS.ssqe_min = ssqe_min_index;
nSTATS.ssqe_max = ssqe_max_index;
nSTATS.msqe = msqe;
nSTATS.aic = aic;
nSTATS.bic = bic;
nSTATS.ch = ch;
nSTATS.db = db;
nSTATS.dunn = dunn;
nSTATS.fpe = fpe;
nSTATS.mdl = mdl;
nSTATS.sil = sil;

%% END