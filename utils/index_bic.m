function [BIC] = index_bic(DATA,PAR)

% ---  Calculate BIC's index for Clustering ---
%
%   [BIC] = index_bic(DATA,PAR)
%
%   Input:
%       DATA.
%           input = input matrix                     	[p x N]
%       PAR.
%           Cx = clusters centroids (prototypes)        [p x Nk]
%           SSE = Sum of Squared Errors for each epoch 	[1 x Nep]
%   Output:
%       BIC = BIC index                                 [cte]

%% INITIALIZATIONS

% Load Data

X = DATA.input;
[p,N] = size(X);

% Load Parameters

[~,Nk] = size(PAR.Cx);
RSS = PAR.SSE(end);

%% ALGORITHM

k = p * Nk;
BIC = N * log(RSS/N) + k * log(N);

%% FILL OUTPUT STRUCTURE

% Dont Need. Output is just a constant.

%% END