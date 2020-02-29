function [OUT] = wta_classify(DATA,PAR)

% --- WTA classifier based test function ---
%
%   [OUT] = wta_classify(DATA,PAR)
% 
%   Input:
%       DATA.
%           input = input matrix                    [p x N]
%       PAR.
%           Cx = prototypes                        	[p x Nk]
%           Cy = class of each prototype/neuron     [Nc x Nk]
%           K = number of nearest neighbors      	[cte]
%           dist = type of distance                	[cte]
%               0: Dot product
%               inf: Chebyshev distance
%               -inf: Minimum Minkowski distance
%               1: Manhattam (city-block) distance
%               2: Euclidean distance
%           Ktype = Kernel Type                  	[cte]
%               = 0 -> non-kernelized algorithm
%   Output:
%       OUT.
%           y_h = classifier's output           	[Nc x N]

%% ALGORITHM

[OUT] = prototypes_class(DATA,PAR);

%% END