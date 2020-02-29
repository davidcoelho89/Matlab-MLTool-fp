function [OUT] = som_classify(DATA,PAR)

% --- SOM classifier based test function ---
%
%   [OUT] = som_classify(DATA,PAR)
%
%   Input:
%       DATA.
%           input = input matrix                 	[p x N]
%       PAR.
%           Cx = prototypes' attributes            	[p x Nk(1) x ... x Nk(Nd)]
%           Cy = prototypes' labels                 [Nc x Nk(1) x ... x Nk(Nd)]
%           K = number of nearest neighbors        	[cte]
%           dist = type of distance                	[cte]
%               0: Dot product
%               inf: Chebyshev distance
%               -inf: Minimum Minkowski distance
%               1: Manhattam (city-block) distance
%               2: Euclidean distance
%           Ktype = kernel Type                  	[cte]
%               = 0 -> non-kernelized algorithm
%   Output:
%       OUT.
%           y_h = classifier's output           	[Nc x N]

%% ALGORITHM

[OUT] = prototypes_class(DATA,PAR);

%% END