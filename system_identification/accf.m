function Px = accf(x,TAU)

% --- AutoCorrelation Coeficient Function ---
%
%   Px = accf(x,TAU)
%
%   Input:
%       x = Sampled signal                    	[1 x Ns]
%       TAU = number of lags to be calculated	[cte]
%   Output:
%       Px = autocorrelation coefficients       [1 x TAU+1]

%% INITIALIZATIONS

x = x(:);               % Force signal to be a column vector
N = length(x);          % Get number of samples
Rx = zeros(TAU+1,1);    % Init accf vector

%% ALGORITHM

% Using Vector Notation

for tau = 0:TAU
    Rx(tau+1) = x(1:N-tau)'*x(tau+1:N)/(N-tau);
end

Px = Rx/var(x);

% Using For Loop

% for tau = 0:TAU
% 	soma = 0;
%     % Backward                  % Forward
% 	for t = tau+1:N             % for t = 1:N-tau
%         aux = x(t-tau)*x(t);  %       aux = x(t)*x(t+tau);
%         soma = soma+aux;
% 	end
% 	Rx(tau+1) = soma/(N-tau);
% end
% 
% Px = Rx/var(x);

%% END