function [PARout] = som_label(DATA,OUT_CL)

% --- SOM Labeling Function ---
%
%   [PARout] = som_label(DATA,OUT_CL)
% 
%   Input:
%       DATA.
%           input = input matrix                                [p x Ntr]
%           output = output matrix                              [Nc x Ntr]
%       OUT_CL.
%           Cx = cluster prototypes                             [p x Nk]
%           dist = type of distance                             [cte]
%               0: Dot product
%               inf: Chebyshev distance
%               -inf: Minimum Minkowski distance
%               1: Manhattam (city-block) distance
%               2: Euclidean distance
%               >2: Minkowsky distance
%           ind = cluster index for each sample                 [1 x Ntr]
%           lbl = type of labeling                              [cte]
%               1: Majority voting
%               2: Average distance
%               3: Minimum distance
%           Ktype = Kernel Type                                 [cte]
%               = 0 -> non-kernelized algorithm
%   Output:
%       PARout.
%           Cx = clusters prototypes                            [p x Nk]
%           Cy = class of each prototype                        [Nc x Nk]

%% ALGORITHM

[PARout] = prototypes_label(DATA,OUT_CL);

%% END