function [OUT] = ts_optm(TYP,BOU,PAR)

% --- Tabu Search for Optimization ---
%
%   [OUT] = ts_optm(TYP,BOU,PAR)
%
%   Input:
%       TYP: Attributes' type.
%           = 1 (integer)
%           = 2 (analog)
%           = 3 (binary)
%       BOU: Bounds for each attribute
%       PAR: parameters' names
%   Output:
%       OUT.
%           x

%% INIT

% ToDo - get components' names of each struct and set values

%% ALGORITHM

% ToDo - All

%% FILL OUTPUT STRUCTURE

OUT = TYP + BOU + PAR;

%% END