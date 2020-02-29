function [MSE] = prototypes_mse(C,DATA,PAR)

% --- Calculate the Mean Squared errors of prototypes ---
%
%   [MSE] = prototypes_mse(C,DATA,PAR)
%
%   Input:
%       C = prototypes [p x k]
%       DATA.
%           dados = input matrix [p x N]
%       PAR.
%           dist = Type of distance 
%               0: dot product
%               2: euclidean distance
%   Output:
%       MSE = mean of squared errors between prototypes and data

%% INITIALIZATION

% Load Data

dados = DATA.dados;
[~,N] = size(dados);

%% ALGORITHM

[SSE] = prototypes_sse(C,DATA,PAR);
MSE = SSE/N;

%% FILL OUTPUT STRUCTURE

% Dont need
    
%% END