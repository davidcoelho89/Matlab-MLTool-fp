function coef = ar_yw(x,p)

% --- Yule-Walker Method for AR(p) Parameters Estimation ---
%
%   Px = accf(x,TAU)
%
%   Input:
%       x = Sampled signal                    	[1 x Ns]
%       p = number of lags to be calculated     [cte]
%   Output:
%       coef = AR coefficients                  [1 x p]

%% ALGORITHM

% TAUmax = 50;            % Maximo TAU
% Px = accf(x,TAUmax);    % Calcula FCAC [r(0) ... r(p)]
% z = Px(1:p+1);          % Primeiros p+1 valores da FCAC: r(0),...,r(p)
% r = z(2:end);           % Ultimos p valores de z: r(1),...,r(p)
% R = toeplitz(z(1:p));   % Matriz de correlacao R
% coef = R\r;             % Estima parametros

Px = accf(x,p);         % Calcula FCAC [r(0) ... r(p)]
r = Px(2:end);       	% Ultimos p valores de z: r(1),...,r(p)
R = toeplitz(Px(1:p));	% Matriz de correlacao R
coef = R\r;             % Estima parametros


%% END