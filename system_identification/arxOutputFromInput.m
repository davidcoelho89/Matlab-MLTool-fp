function y_ts = arxOutputFromInput(u_ts,y_coefs,u_coefs,noise_var)

% --- Generate Output Signal from Input signal and ARX Coefficients ---
%
%   y_ts = arxOutputFromInput(u_ts,y_coefs,u_coefs)
%
%   Input:
%       u_ts = input vector                     [N x 1]
%       y_coefs = output coefficients           [y_lag x 1]
%       u_coefs = input coefficients            [u_lag x 1]
%       noise_var
%   Output:
%       y = Output vector                       [N x 1]

%% INIT

lag_y = length(y_coefs);
lag_u = length(u_coefs);

y_mem = zeros(lag_y,1);
u_mem = zeros(lag_u,1);

N = length(u_ts);
y_ts = zeros(N,1);

if (nargin == 3)
    v = zeros(N,1);
else
    noise_std = sqrt(noise_var);
    v = noise_std*randn(N,1);
end

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