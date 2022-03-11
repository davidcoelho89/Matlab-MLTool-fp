function y_ts_noisy = addTimeSeriesOutilers(y_ts,outlier_ratio,outlier_ext)

% --- Add Outliers to a Time Series ---
%
%   y_ts_noisy = addTimeSeriesOutilers(y_ts,outlier_ratio,outlier_ext)
%
%   Input:
%       y_ts = time series                                      [N x 1]
%       outlier_ratio = how many samples will be contaminated 	[0-1]
%       outlier_ext = until which sample can be contaminated  	[0-1]
%   Output:
%       y_ts_noisy = time series with outliers                  [N x 1]

%% INIT

if(nargin == 2)
    outlier_ext = 0.5;
end

N = length(y_ts);
Nnoise = floor(outlier_ext*N);
y_ts_noisy = y_ts;

y_ts_max = abs(max(y_ts));
y_ts_min = abs(min(y_ts));
max_value = max(y_ts_min,y_ts_max);

signal = sign(randn(Nnoise,1));
noise_prob = rand(Nnoise,1);

%% ALGORITHM

for n = 1:Nnoise
    if (noise_prob(n) <= outlier_ratio)
        y_ts_noisy(n) = y_ts_noisy(n) + signal(n)*3*max_value;
    end
end

%% END