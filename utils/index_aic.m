function [AIC] = index_aic(DATA,PAR)

% ---  Calculate Akaike's index for Clustering ---
%
%   [AIC] = index_aic(DATA,PAR)
%
%   Input:
%       DATA.
%           input = input matrix                        [p x N]
%       PAR.
%           Cx = clusters centroids (prototypes)        [p x Nk]
%           SSE = Sum of Squared Errors for each epoch	[1 x Nep]
%   Output:
%       AIC = AIC index                                 [cte]

%% INITIALIZATIONS

% Load Data

X = DATA.input;
[p,N] = size(X);

% Load Parameters

[~,Nk] = size(PAR.Cx);
RSS = PAR.SSE(end);

%% ALGORITHM

k = p * Nk;
AIC = N * log(RSS/N) + 2 * k;

%% FILL OUTPUT STRUCTURE

% Dont Need. Output is just a constant.

%% END