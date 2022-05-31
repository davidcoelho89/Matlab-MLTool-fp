function [DATA] = buildRegressionMatrices(datasetTS,output_lags,input_lags)

% --- Build Regression Matrix from Time Series or System Signals ---
%
%   [DATA] = buildRegressionMatrices(datasetTS,output_lags,input_lags)
%
%   Input:
%       datasetTS.
%           input = input time series       [Nu x N]
%           output = output time series     [Ny x N]
%       output_lags                         [1 x Nu]
%       input lags                          [1 x Ny]
%   Output:
%       input = Regression Matrix           [lag_u + lag_y x Ns-lag_max]
%       output = Outputs vector             [Ny x Ns-lag_max]
%       lag_input = lag for each input      [1 x Nu]
%       lag_output = lag for each output    [1 x Ny]

%% INITIALIZATIONS

% Get Output Signals

y_ts = datasetTS.output;
[Ny,~] = size(y_ts);

% Get Input Signals

if (isfield(datasetTS,'input'))
    u_ts = datasetTS.input;
    [Nu,~] = size(u_ts);
else
    Nu = 0;
end

% Set Default lags

if (nargin == 1)
    output_lags = 2*ones(1,Ny);
    if (Nu ~= 0)
        input_lags = 2*ones(1,Nu);
    end
elseif (nargin == 2)
    if (Nu ~= 0)
        input_lags = 2*ones(1,Nu);
    end
end

% Verify and Correct lags' sizes

length_lag_y = length(output_lags);
if(length_lag_y > Ny)
    output_lags = output_lags(1,1:Ny);
elseif (length_lag_y < Ny)
    output_lags = [output_lags, 2*ones(1,Ny-length_lag_y)];
end

if (Nu == 0)
    input_lags = [];
else
    length_lag_u = length(input_lags);
    if(length_lag_u > Nu)
        input_lags = input_lags(1,1:Nu);
    elseif(length_lag_u < Nu)
        input_lags = [input_lags, 2*ones(1,Nu-length_lag_u)];
    end
    
end

%% ALGORITHM

if (Nu == 0)
    [X,y] = regressionMatrixFromTimeSeries(y_ts,output_lags);
else
    [X,y] = regressionMatrixFromSystemSignals(y_ts,u_ts,output_lags,input_lags);
end

%% FILL STRUCTURE

DATA.input = X';
DATA.output = y';
DATA.lag_input = input_lags;
DATA.lag_output = output_lags;

%% END