function y_ts_noisy = addTimeSeriesOutilers(y_ts,HP)

% --- Add Outliers to Time Series ---
%
%   y_ts_noisy = addTimeSeriesOutilers(y_ts,outlier_ratio,outlier_ext)
%
%   Input:
%       y_ts = Matrix with time series                            [Ny x N]
%       HP.
%           outlier_rate = rate of samples that will be contaminated [0-1]
%           outlier_ext = until which sample can be contaminated     [0-1]
%   Output:
%       y_ts_noisy = time series with outliers                    [Ny x N]

%% SET DEFAULT OPTIONS

if(nargin == 1 || (isempty(HP)))
    HP.outlier_rate = 0.05;
    HP.outlier_ext = 0.5;
else
    if (~(isfield(HP,'outlier_rate')))
        HP.outlier_rate = 0.05;
    end
    if (~(isfield(HP,'outlier_ext')))
        HP.outlier_ext = 0.5;
    end
end

%% INIT

% Get hyperparameters
outlier_rate = HP.outlier_rate;
outlier_ext = HP.outlier_ext;

% Get number of time series and their extension
[Ny,N] = size(y_ts);

% Until which sample can be contaminated 
Neff = floor(outlier_ext*N); 

% Number of samples that will be contaminated  
Noutlier_samples = floor(Neff*outlier_rate); 

% Init output
y_ts_noisy = y_ts;

%% ALGORITHM

for i = 1:Ny
    % Define outlier value
    y_ts_max = abs(max(y_ts(i,:)));
    y_ts_min = abs(min(y_ts(i,:)));
    max_value = 3*max(y_ts_min,y_ts_max);
    % Define outlier signal
    signal = sign(randn(1,Noutlier_samples));
    % Define samples to be corrupted
    I = randperm(Neff);
    outlier_samples = I(1:Noutlier_samples);
    % Update time series
    y_ts_noisy(i,outlier_samples) = y_ts(i,outlier_samples) + signal.*max_value;
end

%% END