function [X,y] = regressionMatrixFromSISO(y_ts,u_ts,lag_y,lag_u)

% --- Build Regression Matrix from SISO System ---
%
%   [X,y] = regressionMatrixFromSISO(y_ts,u_ts,lag_y,lag_u)
%
%   Input:
%       u_ts = input time series    [Ns x 1]
%       y_ts = output time series   [Ns x 1]
%       lag_y = output lags         [cte]
%       lag_u = input lags          [cte]
%   Output:
%       X = Regression Matrix       [Ns-lag_y x lag_u + lag_y]
%       y = Outputs vector        	[Ns-lag_y x 1]

%% INIT

if (nargin == 2)
    lag_y = 2;
    lag_u = 2;
end

if (nargin == 3)
    lag_u = lag_y;
end

%% ALGORITHM

lag_eff = max([lag_u,lag_y]);
[X_yts,y] = regressionMatrixFromTS(y_ts,lag_eff);
[X_uts,~] = regressionMatrixFromTS(u_ts,lag_eff);

if (lag_y == lag_eff)
    X_yts_eff = X_yts;
    X_uts_eff = X_uts(:,1:lag_u);
else
    X_yts_eff = X_yts(:,1:lag_y);
    X_uts_eff = X_uts;
end

X = [X_yts_eff, X_uts_eff];

%% END