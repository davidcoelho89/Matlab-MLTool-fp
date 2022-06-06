function [STATS] = regress_stats_1turn(DATA,OUT)

% --- Provide Statistics of 1 turn of Regression ---
%
%   [STATS] = regress_stats_1turn(OUT,OUT_h)
% 
%    	DATA.
%           output = actual outputs             [No x N]
%     	OUT.
%           y_h = estimated outputs             [No x N]
%   Output:
%       STATS.
%       	err = error Matrix                  [No x N]
%           rmse = root mean squared error      [No x 1]

%% INITIALIZATIONS

y = DATA.output;
y_h = OUT.y_h;

[~,N] = size(y);

%% ALGORITHM

err = y - y_h;
rmse = sqrt((1/N)*sum(err.^2));

%% FILL OUTPUT STRUCTURE

STATS.err = err;
STATS.rmse = rmse;

%% THEORY



%% END