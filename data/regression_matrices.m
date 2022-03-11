function [DATA] = regression_matrices(DATAts,OPT)

% --- Build Regression Matrix from ARX System ---
%
%   [DATA] = regression_matrices(DATAts,OPT)
%
%   Input:
%       DATA.
%           input = input time series       [Nu x N]
%           output = output time series     [Ny x N]
%       OPT.
%           lag_y = output lags             [Nu x 1]
%           lag_u = input lags              [Ny x 1]
%   Output:
%       input = Regression Matrix       [Ns-lag_y x lag_u + lag_y]
%       output = Outputs vector        	[Ns-lag_y x 1]

%% INIT

if (nargin == 1)
    OPT.lag_y = 2;
    OPT.lag_u = 2;
end

u_ts = DATAts.input;
y_ts = DATAts.output;

lag_u = OPT.lag_u;
lag_y = OPT.lag_y;

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

%% FILL STRUCTURE

DATA.input = X;
DATA.output = y;

%% END