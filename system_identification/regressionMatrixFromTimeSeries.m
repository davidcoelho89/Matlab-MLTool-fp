function [X,y] = regressionMatrixFromTimeSeries(y_ts,lag_y)

% --- Build Regression Matrix from Time Series (AR Model) ---
%
%   [X,y] = regressionMatrixFromTimeSeries(y_ts,lag_y)
%
%   Input:
%       y_ts = vector with signal samples   [1 x Ns]
%       lag_y = AR model order             	[cte]
%   Output:
%       X = Regression Matrix             	[Ns-p x p]
%       y = Outputs vector                 	[Ns-p x 1]

%% INIT

% Force signal to column-vector
y_ts = y_ts(:);

% Numer of samples
Ns = length(y_ts);

% Input Matrix
X = zeros(Ns-lag_y, lag_y);

%% ALGORITHM

for col = 1:lag_y
    X(:,col) = y_ts(lag_y-col+1:end-col);
end

y = y_ts(lag_y+1:end);

%% END