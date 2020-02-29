function [PARout] = ols_train(DATA,PAR)

% --- OLS classifier training ---
%
%   [PARout] = ols_train(DATA,PAR)
%
%   Input:
%       DATA.
%           input = input matrix                        [p x N]
%           output = output matrix                      [Nc x N]
%       PAR.
%           aprox = type of approximation               [cte]
%               1 -> W = Y*pinv(X);
%               2 -> W = Y*X'/(X*X');
%               3 -> W = Y/X;
%   Output:
%       PARout.
%           W = Regression / Classification Matrix      [Nc x p+1]

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(PAR))),
    PARaux.aprox = 1;       % Neurons' labeling function
    PAR = PARaux;
else
    if (~(isfield(PAR,'aprox'))),
        PAR.aprox = 1;
    end
end

%% INITIALIZATIONS

% Data matrix
X = DATA.input;         % Input matrix
Y = DATA.output;        % Output matrix

% Hyperparameters Init
aprox = PAR.aprox;      % Type of approximation

% Problem Init
[~,N] = size(Y);        % Number of samples

% add bias to input matrix [x0 = +1]
X = [ones(1,N) ; X];	

%% ALGORITHM

if aprox == 1,
    W = Y*pinv(X);
elseif aprox == 2,
    W = Y*X'/(X*X');
elseif aprox == 3,
    W = Y/X;
else
    disp('type a valid option: 1, 2 or 3');
    W = [];
end

%% FILL OUTPUT STRUCTURE

PARout = PAR;
PARout.W = W;

%% END