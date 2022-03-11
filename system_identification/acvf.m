function Cx = acvf(x,TAU)

% --- AutoCovariance Function ---
%
%   Cx = acvf(x,TAU)
%
%   Input:
%       x = Sampled signal                    	[1 x Ns]
%       TAU = number of lags to be calculated	[cte]
%   Output:
%       Cx = autocovariance coefficients        [1 x TAU+1]

%% INITIALIZATIONS

x = x(:);               % Force signal to be a column vector
N = length(x);          % Get number of samples
Rx = zeros(TAU+1,1);	% Init acf vector

%% ALGORITHM

% Using Vector Notation

for tau = 0:TAU
    Rx(tau+1) = x(1:N-tau)'*x(tau+1:N)/(N-tau);
end

Cx = Rx - mean(x)^2;

% Using For Loop

% for tau = 0:TAU
%     soma = 0;
%     % Forward                   % Backward
%     for t = 1:N-tau             % for t = tau+1:N
%         aux = y(t)*y(t+tau);    %     aux = y(t-tau)*y(t);
%         soma = soma+aux;
%     end
%     Rx(tau+1) = soma/(N-tau);
% end
%
% Cx = Rx - mean(x)^2;

%% END