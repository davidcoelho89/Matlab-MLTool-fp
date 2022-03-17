function y_ts_noisy = addTimeSeriesNoise(y_ts,HP)

% --- Add noise to Time Series ---
%
%   y_ts_noisy = addTimeSeriesNoise(y_ts,noise_var)
%
%   Input:
%       y_ts = Matrix with time series                          [Ny x N]
%       HP.
%           noise_var = Variance of gaussian noise to be added  [cte]
%           noise_mean = Mean of gaussian noise to be added     [cte]
%   Output:
%       y_ts_noisy = Cell with all signals (noise added)        [Ny x N]

%% SET DEFAULT OPTIONS

if(nargin == 1 || (isempty(HP)))
    HP.noise_var = 0.01;
    HP.noise_mean = 0;
else
    if (~(isfield(HP,'noise_var')))
        HP.noise_var = 0.01;
    end
    if (~(isfield(HP,'noise_mean')))
        HP.noise_mean = 0;
    end
end

%% INIT

% Get Hyperparameters
noise_var = HP.noise_var;
noise_mean = HP.noise_mean;
noise_std = sqrt(noise_var);

% Init Output
[Ny,N] = size(y_ts);
y_ts_noisy = zeros(Ny,N);

%% ALGORITHM

for i = 1:Ny
    y_ts_noisy(i,:) = y_ts(i,:) + noise_mean + noise_std*randn(1,N);
end

%% END