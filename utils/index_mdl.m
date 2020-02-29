function [MDL] = index_mdl(DATA,PAR)

% ---  Calculate MDL's index for Clustering ---
%
%   [MDL] = mdl_index(DATA,PAR)
%
%   Input:
%       DATA.
%           input = input matrix                     	[p x N]
%       PAR.
%           Cx = clusters centroids (prototypes)        [p x Nk]
%           ind = cluster index for each sample         [1 x N]
%           SSE = Sum of Squared Errors for each epoch 	[1 x Nep]
%   Output:
%       MDL = MDL index                                 [cte]

%% INIT

% Load Data

X = DATA.input;
[p,N] = size(X);

% Load Parameters

Cx = PAR.Cx;
[~,Nk] = size(Cx);
RSS = PAR.SSE(end);

%% ALGORITHM

k = p*Nk;
MDL = N*log(RSS/N) + (k/2)*log(N);

%% FILL OUTPUT STRUCTURE

% Dont Need. Output is just a constant.

%% END