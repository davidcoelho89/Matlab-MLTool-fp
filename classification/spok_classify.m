function [OUT] = spok_classify(DATA,PAR)

% --- SParse Online adptive Kernel Classify Function ---
%
%   [OUT] = spok_classify(DATA,PAR)
% 
%   Input:
%       DATA.
%           input = input matrix                                [p x N]
%       PAR.
%           Cx = prototypes' attributes                         [p x Nk]
%           Cy = prototypes' labels                             [Nc x Nk]
%           K = number of nearest neighbors                     [cte]
%           knn_type = type of knn aproximation                 [cte]
%               1: Majority Voting
%               2: Weighted KNN
%           dist = type of distance (if Ktype = 0)              [cte]
%               0: Dot product
%               inf: Chebyshev distance
%               -inf: Minimum Minkowski distance
%               1: Manhattam (city-block) distance  
%               2: Euclidean distance
%           Ktype = kernel type ( see kernel_func() )           [cte]
%           sigma = kernel hyperparameter ( see kernel_func() ) [cte]
%           alpha = kernel hyperparameter ( see kernel_func() ) [cte]
%           theta = kernel hyperparameter ( see kernel_func() ) [cte]
%           gamma = kernel hyperparameter ( see kernel_func() ) [cte]
%   Output:
%       OUT.
%           y_h = classifier's output                           [Nc x N]
%           win = closest prototype to each sample              [1 x N]
%           dist = distance of sample from each prototype       [Nk x N]
%           near_ind = indexes for nearest prototypes           [K x N]

%% ALGORITHM

[OUT] = prototypes_class(DATA,PAR);

%% END