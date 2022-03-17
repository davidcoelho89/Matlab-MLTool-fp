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
%           lag_u = output lags             [1 x Nu]
%           lag_y = input lags              [1 x Ny]
%   Output:
%       input = Regression Matrix           [Ns-lag_max x lag_u + lag_y]
%       output = Outputs vector             [Ns-lag_max x Ny]
%       lag_input = lag for each input      [1 x Nu]
%       lag_output = lag for each output    [1 x Ny]

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
    if (Nu ~= 0)
        OPT.lag_u = 2*ones(1,Nu);
    end
else
    if (~(isfield(OPT,'lag_y')))
        OPT.lag_y = 2*ones(1,Ny);
    end
    if (~(isfield(OPT,'lag_u')))
        if (Nu ~= 0)
            OPT.lag_u = 2*ones(1,Nu);
        end
    end
end

length_lag_y = length(OPT.lag_y);

if(length_lag_y > Ny)
    OPT.lag_y = OPT.lag_y(1,1:Ny);
elseif (length_lag_y < Ny)
    OPT.lag_y = [OPT.lag_y, 2*ones(1,Ny-length_lag_y)];
end

if (Nu == 0)
    OPT.lag_u = [];
else
    length_lag_u = length(OPT.lag_u);
    if(length_lag_u > Nu)
        OPT.lag_u = OPT.lag_u(1,1:Nu);
    elseif(length_lag_u < Nu)
        OPT.lag_u = [OPT.lag_u, 2*ones(1,Nu-length_lag_u)];
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
DATA.lag_input = lag_u;
DATA.lag_output = lag_y;

%% END