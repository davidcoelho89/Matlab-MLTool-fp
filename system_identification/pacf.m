function Rx = pacf(x,TAU)

% --- Partial AutoCorrelation Function ---
%
%   Rx = pacf(x,TAU)
%
%   Input:
%       x = Sampled signal                          [1 x Ns]
%       TAU = number of lags to be calculated       [cte]
%   Output:
%       Rx = Partial AutoCorrelation coefficients   [1 x TAU+1]

%% INITIALIZATIONS

x = x(:);               % Force signal to be a column vector
Rx = zeros(TAU,1);      % Init pacf vector

%% ALGORITHM

% Using Vector Notation

for p = 1:TAU
    ah_aryule = aryule(x,p);
    Rx(p) = -ah_aryule(end);
end

%% END