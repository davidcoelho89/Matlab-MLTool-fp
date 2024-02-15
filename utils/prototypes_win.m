function [c] = prototypes_win(C,sample,PAR)

% --- Calculate the closest prototype to a sample ---
%
%   [c] = prototype_win(C,sample,PAR)
%
%   Input:
%       C = prototypes                                          [p x Q]
%       sample = data vector                                    [p x 1]
%       PAR.
%           dist = Type of metric                               [cte]
%               0: dot product
%               inf: Chebyshev distance
%               -inf: Minimum Minkowski distance
%               1: Manhattam (city-block) distance
%               2: Euclidean distance
%               >2: Minkowsky distance
%           Ktype = kernel type ( see kernel_func() )           [cte]
%           sigma = kernel hyperparameter ( see kernel_func() ) [cte]
%           alpha = kernel hyperparameter ( see kernel_func() ) [cte]
%           theta = kernel hyperparameter ( see kernel_func() ) [cte]
%           gamma = kernel hyperparameter ( see kernel_func() ) [cte]
%   Output:
%       c = closest prototype to sample                         [cte]

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 2) || (isempty(PAR)))
    PARaux.dist = 2;
    PARaux.Ktype = 0;
    PAR = PARaux;
else
    if (~(isfield(PAR,'dist')))
        PAR.dist = 2;
    end
    if (~(isfield(PAR,'Ktype')))
        PAR.Ktype = 0;
    end
end

%% INITIALIZATION

% Init Variables

[~,Q] = size(C);                % Number of prototypes
Vdist = zeros(1,Q);             % Vector of distances
dist = PAR.dist;                % Type of distance
Ktype = PAR.Ktype;              % Kernel type

%% ALGORITHM
    
% Calculate Distance Vector

for i = 1:Q 
    % Get Prototype
    prot = C(:,i);
    % Calculate distance
    Vdist(i) = vectors_dist(prot,sample,PAR);
end

% Choose Closest (winner) Prototype

% dot product
if(dist == 0 && Ktype == 0)
    [~,win] = max(Vdist);
% Other distances
else
    [~,win] = min(Vdist);
end

%% FILL OUTPUT STRUCTURE

c = win;

%% END