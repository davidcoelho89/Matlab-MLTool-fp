function y_ts = arxOutputFromInput(u_ts,HP)

% --- Generate Output Signal from Input signal and ARX Coefficients ---
%
%   y_ts = arxOutputFromInput(u_ts,HP)
%
%   Input:
%       u_ts = input vector               	[N x 1]
%       HP.
%           y_coefs = output coefficients	[y_lag x 1]
%           u_coefs = input coefficients  	[u_lag x 1]
%           noise_var = noise variance      [cte]
%           noise_mean = noise mean         [cte]
%   Output:
%       y = Output vector                   [N x 1]

%% SET DEFAULT HYPERPARAMETERS

if(nargin == 0 || (isempty(HP)))
    HP.y_coefs = [0.4,-0.6];
    HP.u_coefs = 2;
    HP.noise_var = 0.01;
    HP.noise_mean = 0;
else
    if (~(isfield(HP,'y_coefs')))
        HP.y_coefs = [0.4,-0.6];
    end
    if (~(isfield(HP,'u_coefs')))
        HP.u_coefs = 2;
    end
    if (~(isfield(HP,'noise_var')))
        HP.noise_var = 0.01;
    end
	if (~(isfield(HP,'noise_mean')))
        HP.noise_mean = 0;
	end
end

%% INIT

% Get Hyperparameters
u_coefs = HP.u_coefs;
y_coefs = HP.y_coefs;
noise_var = HP.noise_var;
noise_mean = HP.noise_mean;

% Define maximum lags
lag_y = length(y_coefs);
lag_u = length(u_coefs);

% Init memories
y_mem = zeros(lag_y,1);
u_mem = zeros(lag_u,1);

% Init output signal
N = length(u_ts);
y_ts = zeros(N,1);

% Build noise signal
noise_std = sqrt(noise_var);
v = noise_mean + noise_std*randn(N,1);

%% ALGORITHM

for n = 1:N
    % Update Output
    y_ts(n) = y_coefs*y_mem + u_coefs*u_mem + v(n);
    
    % Update Input memory
    if(lag_u == 1)
        u_mem = u_ts(n);
    else
        u_mem = [u_ts(n);u_mem(1:end-1)];
    end
    
    % Update Output memory
    if (lag_y == 1)
        y_mem = y_ts(n);
    else
        y_mem = [y_ts(n);y_mem(1:end-1)];
    end
end

%% END