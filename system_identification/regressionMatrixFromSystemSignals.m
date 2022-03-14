function [X,y] = regressionMatrixFromSystemSignals(y_ts,u_ts,lag_y,lag_u)

% --- Build Regression Matrix from System's Signals ---
%
%   [X,y] = regressionMatrixFromSystemSignals(y_ts,u_ts,lag_y,lag_u)
%
%   Input:
%       u_ts = input time series    [Nu x Ns]
%       y_ts = output time series   [Ny x Ns]
%       lag_y = output lags         [1 x Ny]
%       lag_u = input lags          [1 x Nu]
%   Output:
%       X = Regression Matrix       [Ns-lag_max x lag_u + lag_y]
%       y = Outputs vector        	[Ns-lag_max x 1]

%% INIT

[Ny,~] = size(y_ts);
[Nu,~] = size(u_ts);

if (nargin == 2)
    lag_y = 2*ones(1,Nu);
    lag_u = 2*ones(1,Nu);
elseif (nargin == 3)
    lag_u = 2*ones(1,Nu);
end

X = [];
y = [];

%% ALGORITHM

lag_max = max([lag_u,lag_y]);

% Build output and Init Regression Matrix
for i = 1:Ny
    [X_yts,y_out] = regressionMatrixFromTimeSeries(y_ts(i,:),lag_max);
    y = [y,y_out];
    X = [X,X_yts(:,1:lag_y(i))];
end

% Finish Regression Matrix
for i = 1:Nu
    [X_uts,~] = regressionMatrixFromTimeSeries(u_ts(i,:),lag_max);
    X = [X,X_uts(:,1:lag_u(i))];
end

%% END