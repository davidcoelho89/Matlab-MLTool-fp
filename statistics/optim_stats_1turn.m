function [STATS] = optim_stats_1turn(OUT,OUT_h)

% --- Provide Statistics of 1 turn of Optimization ---
%
%   [STATS] = optim_stats_1turn(OUT,OUT_h)
% 
%   Input:
%    	OUT = actual outputs      [y x N]
%     	OUT_h = estimated outputs [y_h x N]
%   Output:
%       STATS.
%       	

%% INITIALIZATIONS

%% ALGORITHM

%% FILL OUTPUT STRUCTURE

STATS = OUT + OUT_h;

%% THEORY

%% END