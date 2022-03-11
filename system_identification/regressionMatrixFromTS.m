function [X,y] = regressionMatrixFromTS(timeSeries,p)

% --- Build Regression Matrix from Time Series ---
%
%   [X,y] = regressionMatrixFromTS(timeSeries,p)
%
%   Input:
%       timeSeries = vector with signal samples [Ns x 1]
%       p = AR model order                      [cte]
%   Output:
%       X = Regression Matrix                   [Ns-p x p]
%       y = Outputs vector                      [Ns-p x 1]

%% INIT

% Force signal to column-vector
timeSeries = timeSeries(:);

% Numer of samples
Ns = length(timeSeries);

% Input Matrix
X = zeros(Ns-p, p);

%% ALGORITHM

for col = 1:p
    X(:,col) = timeSeries(p-col+1:end-col);
end

y = timeSeries(p+1:end);

%% END