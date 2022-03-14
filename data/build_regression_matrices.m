function [DATA] = build_regression_matrices(DATAts,OPT)

% --- Build Regression Matrix from Time Series or System Signals ---
%
%   [DATA] = build_regression_matrices(DATAts,OPT)
%
%   Input:
%       DATAts.
%           input = input time series       [Nu x N]
%           output = output time series     [Ny x N]
%       OPT.
%           lag_y = output lags             [1 x Nu]
%           lag_u = input lags              [1 x Ny]
%   Output:
%       input = Regression Matrix       [Ns-lag_max x lag_u + lag_y]
%       output = Outputs vector        	[Ns-lag_max x 1]

%% INITIALIZATIONS

% Get Output Signals
y_ts = DATAts.output;
[Ny,~] = size(y_ts);

% Get Input Signals
if (isfield(DATAts,'input'))
    u_ts = DATAts.input;
    [Nu,~] = size(u_ts);
else
    Nu = 0;
end

% Set Default lags
if (nargin == 1|| (isempty(OPT)))
    OPT.lag_y = 2*ones(1,Ny);
    if (Nu == 0)
        OPT.lag_u = [];
    else
        OPT.lag_u = 2*ones(1,Nu);
    end
else
    if (~(isfield(OPT,'lag_y')))
        OPT.lag_y = 2*ones(1,Ny);
    end
    if (~(isfield(OPT,'lag_u')))
        if (Nu == 0)
            OPT.lag_u = [];
        else
            OPT.lag_u = 2*ones(1,Nu);
        end
    end
end

% Get lags and their lengths
lag_u = OPT.lag_u;
lag_y = OPT.lag_y;

%% ALGORITHM

if (Nu == 0)
    [X,y] = regressionMatrixFromTimeSeries(y_ts,lag_y);
else
    [X,y] = regressionMatrixFromSystemSignals(y_ts,u_ts,lag_y,lag_u);
end

%% FILL STRUCTURE

DATA.input = X;
DATA.output = y;

%% END