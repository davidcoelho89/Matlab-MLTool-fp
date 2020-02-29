function [PARout] = knn_train(DATA,PAR)

% --- KNN classifier training ---
%
%   [PARout] = knn_train(DATA,PAR)
%
%   Input:
%       DATA.
%           input = attributes matrix               [p x N]
%           output = labels matrix                  [Nc x N]
%       PAR.
%           dist = type of distance             	[cte]
%               0: Dot product
%               inf: Chebyshev distance
%               -inf: Minimum Minkowski distance
%               1: Manhattam (city-block) distance
%               2: Euclidean distance%   Output:
%           Ktype = Kernel Type                     [cte]
%               = 0 -> non-kernelized algorithm
%           K = number of nearest neighbors         [cte]
%       PARout.
%           Cx = prototypes' attributes             [p x N]
%           Cy = prototypes' labels                	[Nc x N]

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(PAR))),
    PARaux.dist = 02;      	% type of distance
    PARaux.Ktype = 0;       % Non-kernelized Algorithm
    PARaux.K = 3;           % Number of nearest neighbors
    PAR = PARaux;
else
    if (~(isfield(PAR,'dist'))),
        PAR.dist = 02;
    end
    if (~(isfield(PAR,'Ktype'))),
        PAR.Ktype = 0;
    end
    if (~(isfield(PAR,'K'))),
        PAR.K = 3;
    end
end

%% ALGORITHM

PARout = PAR;
PARout.Cx = DATA.input;
PARout.Cy = DATA.output;

%% END