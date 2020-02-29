function [OUT] = knn_classify(DATA,PAR)

% --- KNN classifier based test function ---
%
%   [OUT] = knn_classify(DATA,PAR)
%
%   Input:
%       DATA.
%           input = attributes matrix               [p x N]
%       PAR.
%           Cx = prototypes' attributes             [p x Nk]
%           Cy = prototypes' labels                 [Nc x Nk]
%           dist = type of distance                	[cte]
%               0: Dot product
%               inf: Chebyshev distance
%               -inf: Minimum Minkowski distance
%               1: Manhattam (city-block) distance
%               2: Euclidean distance
%           Ktype = Kernel Type                  	[cte]
%               = 0 -> non-kernelized algorithm
%           K = number of nearest neighbors      	[cte]
%   Output:
%       OUT.
%           y_h = classifier's output               [Nc x N]

%% ALGORITHM

[OUT] = prototypes_class(DATA,PAR);

%% END