function [FPE] = index_fpe(DATA,PAR)

% ---  Calculate Akaike's Final Prediction Error index for Clustering ---
%
%   [FPE] = index_fpe(DATA,PAR)
%
%   Input:
%       DATA.
%           input = input matrix                     	[p x N]
%       PAR.
%           Cx = clusters centroids (prototypes)        [p x Nk]
%           ind = cluster index for each sample         [1 x N]
%           SSE = Sum of Squared Errors for each epoch 	[1 x Nep]
%   Output:
%       FPE = FPE index                                 [cte]

%% INIT

% Load Data

X = DATA.input;
[p,N] = size(X);

% Load Parameters

RSS = PAR.SSE(end);
[~,Nk] = size(PAR.Cx);

%% ALGORITHM

k = p * Nk;
FPE = N * log(RSS/N) + N * log((N+k)/(N-k));

%% FILL OUTPUT STRUCTURE

% Dont Need. Output is just a constant.

%% END