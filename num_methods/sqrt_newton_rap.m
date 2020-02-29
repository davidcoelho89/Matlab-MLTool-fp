function [x] = sqrt_newton_rap(r)

% --- Newton Raphson Method for Squared Root ---
%
%   [x] = sqrt_newton_rap(r,n)
%
%   Input:
%       r = number whose squared root is needed
%   Output:
%       x = squared root of r

%% INITIALIZATIONS

er = 10e-4; % stop criterion

%% ALGORITHM

% Definition of x0

x = 0;      
while(x*x - r < 0),
    x = x+1;
end

x_sup = x;
x_inf = x-1;

x = (x_sup+x_inf)/2;

% Recursive Method

while 1 == 1,
    % Calculate next step
    x = x - (x*x - r)/(2*x);
    % Stop criterion
    if abs(x*x - r) < er,
        break;
    end
end

%% THEORY

% Newton-Raphson Method
% Used to aproximate roots of "f(x) = 0" functions

% When dealing with squared root: wants to find "x" that is root of "r"
% So: r = x^2   ->  x^2 - r = 0 ->  f(x) = x^2 - r

% Method
% Define initial value for x (x0)
% Apply the recursive method:
%   x(k+1) = x(k) - fx(k)/f'x(k)

%% END