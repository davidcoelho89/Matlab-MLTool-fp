function [STATS] = regress_stats_1turn(DATA,OUT)

% --- Provide Statistics of 1 turn of Regression ---
%
%   [STATS] = regress_stats_1turn(OUT,OUT_h)
% 
%    	DATA.
%           output = actual outputs             [1 x N] or [Nc x N]
%     	OUT.
%           y_h = estimated outputs             [1 x N] or [Nc x N]
%   Output:
%       STATS.
%       	err = error vector                  [1 x N]
%           rmse = root mean squared error      [cte]

%% INITIALIZATIONS

y = DATA.output;
y_h = OUT.y_h;

N = length(y);

%% ALGORITHM

err = y - y_h;
rmse = sqrt((1/N)*sum(err.^2));

%% FILL OUTPUT STRUCTURE

STATS.err = err;
STATS.rmse = rmse;

%% THEORY



%% END