function [PARout] = ksom_ps_label(DATA,OUT_CL)

% --- KSOM-PS Labeling Function ---
%
%   [PARout] = ksom_ps_label(DATA,OUT_CL)
% 
%   Input:
%       DATA.
%           input = input matrix                                [p x Ntr]
%           output = output matrix                              [Nc x Ntr]
%       OUT_CL.
%           Cx = cluster prototypes         [p x Nk(1) x Nk(2) x ... x Nk(Nd)]
%           Mphi = mapped input matrix                          [m x Ntr]
%           dist = type of distance                             [cte]
%               0: Dot product
%               inf: Chebyshev distance
%               -inf: Minimum Minkowski distance
%               1: Manhattam (city-block) distance
%               2: Euclidean distance
%           ind = cluster index for each sample                 [Nd x Ntr]
%           lbl = type of labeling                              [cte]
%               1: Majority voting
%               2: Average distance
%               3: Minimum distance
%           Ktype = kernel type ( see kernel_func() )           [cte]
%           sigma = kernel hyperparameter ( see kernel_func() ) [cte]
%           order = kernel hyperparameter ( see kernel_func() ) [cte]
%           alpha = kernel hyperparameter ( see kernel_func() ) [cte]
%           theta = kernel hyperparameter ( see kernel_func() ) [cte]
%           gamma = kernel hyperparameter ( see kernel_func() ) [cte]
%   Output:
%       PARout.
%           Cx = clusters prototypes        [m x Nk(1) x Nk(2) x ... x Nk(Nd)]
%           Cy = class of each prototype    [Nc x Nk(1) x Nk(2) x ... Nx k(Nd)]

%% INITIALIZATION

% Get mapped input data
DATA.input = OUT_CL.Mphi;

% Hold Kernel Type
Ktype = OUT_CL.Ktype;

% Work in data space
OUT_CL.Ktype = 0;

%% ALGORITHM

[PARout] = prototypes_label(DATA,OUT_CL);

PARout.Ktype = Ktype;

%% END