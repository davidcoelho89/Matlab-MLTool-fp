function [c] = prototypes_win(C,sample,PAR)

% --- Calculate the closest prototype to a sample ---
%
%   [c] = prototype_win(C,sample,PAR)
%
%   Input:
%       C = prototypes                                          [p x Nk]
%       sample = data vector                                    [p x 1]
%       PAR.
%           dist = Type of metric                               [cte]
%               0: dot product
%               inf: Chebyshev distance
%               -inf: Minimum Minkowski distance
%               1: Manhattam (city-block) distance
%               2: Euclidean distance
%           Ktype = kernel type ( see kernel_func() )           [cte]
%           sigma = kernel hyperparameter ( see kernel_func() ) [cte]
%           order = kernel hyperparameter ( see kernel_func() ) [cte]
%           alpha = kernel hyperparameter ( see kernel_func() ) [cte]
%           theta = kernel hyperparameter ( see kernel_func() ) [cte]
%           gamma = kernel hyperparameter ( see kernel_func() ) [cte]
%   Output:
%       c = closest prototype to sample                         [cte]

%% INITIALIZATION

% Init Variables

[~,Nk] = size(C);               % Number of prototypes
Vdist = zeros(1,Nk);            % Vector of distances
if (~(isfield(PAR,'dist'))),    % Type of distance
    dist = 2;
else
    dist = PAR.dist;
end

%% ALGORITHM
    
% Calculate Distance Vector

for i = 1:Nk,
    % Get Prototype
    prot = C(:,i);
    % Calculate distance
    Vdist(i) = vectors_dist(prot,sample,PAR);
end

% Choose Closest (winner) Prototype

% dot product
if(dist == 0)
    [~,win] = max(Vdist);
% Other distances
else
    [~,win] = min(Vdist);
end

%% FILL OUTPUT STRUCTURE

c = win;

%% END