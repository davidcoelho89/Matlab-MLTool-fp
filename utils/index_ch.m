function [CH] = index_ch(DATA,PAR)

% ---  Calculate CH index for Clustering ---
%
%   [CH] = index_ch(DATA,PAR)
%
%   Input:
%       DATA.
%           input = input matrix                     	[p x N]
%       PAR.
%           Cx = clusters centroids (prototypes)        [p x Nk]
%           ind = cluster index for each sample         [1 x N]
%           SSE = Sum of Squared Errors for each epoch 	[1 x Nep]
%   Output:
%       CH = CH index                                   [cte]

%% INITIALIZATIONS

% Load Data

X = DATA.input;
[p,N] = size(X);

% Load Parameters

indexes = PAR.ind;
k = length(find(unique(indexes)));

% Init Clusters

clusters = cell(1,k);
for l = 1:k,
    VJ = X(:,indexes == l);
    clusters{l} = VJ';
end

% Init Aux Variables

M = mean(X,2)';     % Centroid of the whole dataset
Ni = 0;             % accumulates number of samples per cluster
Wq = zeros(p);      % Initial value of within-cluster scatter matrix
Bq = zeros(p);      % Initial value of between-cluster scatter matrix

%% ALGORITHM

if k == 1
    CH = NaN;
else
    for j = 1:k,
        nj = size(clusters{j});	% Number of samples of cluster j
        mj = mean(clusters{j});	% Mean of cluster j
        Cj = clusters{j};       % Get data points from cluster k
        Swj = cov(Cj,1);        % Within-cluster scatter mat of cluster j
        Sbj = (mj-M)'*(mj-M);	% Between-cluster scatter mat of cluster j
        Wq = Wq + nj(1)*Swj;    % 
        Bq = Bq + nj(1)*Sbj;    % 
        Ni = Ni + nj(1);        % 
    end
    
    Wq = Wq/Ni;  % Within-cluster scatter matrix
    Bq = Bq/Ni;  % Between-cluster scatter matrix
    
    CH = (trace(Bq)/(k-1)) / (trace(Wq)/(N-k));
end

%% FILL OUTPUT STRUCTURE

% Dont Need. Output is just a constant.

%% END