function [PARout] = elm_train(DATA,PAR)

% --- ELM classifier training function ---
%
%   [PARout] = elm_train(DATA,PAR)
%
%   Input:
%       DATA.
%           input = attributes                                  [p x N]
%           output = labels                                     [Nc x N]
%       PAR.
%           Nh = number of hidden neurons                       [cte]
%           Nlin = non-linearity                                [cte]
%               1 -> Sigmoid                                    [0 e 1]
%               2 -> Hyperbolic Tangent                         [-1 e +1]
%   Output:
%       PARout.
%           W{1} = Weight Matrix: Inputs to Hidden Layer        [Nh x p+1]
%           W{2} = Weight Matrix: Hidden layer to output layer	[Nc x Nh+1]

%% SET DEFAULT HYPERPARAMETERS

if ((nargin == 1) || (isempty(PAR)))
    PARaux.Nh = 25;         % Number of hidden neurons
    PARaux.Nlin = 2;        % Non-linearity
    PAR = PARaux;
    
else
    if (~(isfield(PAR,'Nh')))
        PAR.Nh = 25;
    end
    if (~(isfield(PAR,'Nlin')))
        PAR.Nlin = 2;
    end
end

%% INITIALIZATION

% Data Initialization
X = DATA.input;                     % Input Matrix
D = DATA.output;                    % Output Matrix

% Hyperparameters Initialization
Nh = PAR.Nh;                        % number of hidden neurons
Nlin = PAR.Nlin;                    % Non-linearity 

% Problem Initialization
[p,N] = size(X);                    % Size of input matrix

% Weight Matrices Initialization

if (isfield(PAR,'W'))
    W{1} = PAR.W{1};            	% if already initialized
else
    W{1} = 0.01*(2*rand(Nh,p+1)-1);	% Weights of Hidden layer [-0.01,0.01]
end

% Add bias to input matrix
X = [ones(1,N) ; X];                % x0 = +1

% Activation matrix of hidden neurons
Xk = zeros(Nh+1,N);

%% ALGORITHM

for t = 1:N

    % HIDDEN LAYER
    xi = X(:,t);              	% Get input sample
    Ui = W{1} * xi;           	% Activation of hidden layer neurons
    Yi = elm_f_ativ(Ui,Nlin);	% Non-linear function

    % OUTPUT LAYER INPUT
    Xk(:,t) = [1; Yi];           % Add bias to activation vector

end

% Output Layer Weight Matrix Calculation (by Pseudo-inverse)

W{2} = D*pinv(Xk);

%% FILL OUTPUT STRUCTURE

PARout = PAR;
PARout.W = W;

%% END