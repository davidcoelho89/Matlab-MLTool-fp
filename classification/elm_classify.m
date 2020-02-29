function [OUT] = elm_classify(DATA,PAR)

% --- ELM classifier test ---
%
%   [OUT] = elm_classify(DATA,PAR)
%
%   Input:
%       DATA.
%           input = attributes [p x N]
%       PAR.
%           W{1} = Weight Matrix: Inputs to Hidden Layer        [Nh x p+1]
%           W{2} = Weight Matrix: Hidden layer to output layer 	[Nc x Nh+1]
%           Nh = number of hidden neurons                       [cte]
%           Nlin = non-linearity                                [cte]
%               1 -> Sigmoid                                    [0 e 1]
%               2 -> Hyperbolic Tangent                         [-1 e +1]
%   Output:
%       OUT.
%           y_h = classifier's output                           [Nc x N]

%% INIT

% Get data
X = DATA.input;             % Input data of the problem

% Get parameters
W{1} = PAR.W{1};          	% Hidden layer weight Matrix
W{2} = PAR.W{2};          	% Output layer weight Matrix
Nlin = PAR.Nlin;            % non-linearity
Nh = PAR.Nh;              	% number of hidden neurons

% Problem Initialization
[~,N] = size(X);            % number of classes and samples

Xk = zeros(Nh+1,N);        	% Activation matrix of hidden neurons

% Add bias to input matrix
X = [ones(1,N);X];          % x0 = +1

%% ALGORITHM

for t = 1:N,
    
    % HIDDEN LAYER
    xi = X(:,t);                % Add bias (x0 = +1) to input vector
    Ui = W{1} * xi;           	% Activation of hidden layer neurons   
    Yi = elm_f_ativ(Ui,Nlin);   % Non-linear function

    % OUTPUT LAYER INPUT
    Xk(:,t) = [+1; Yi];       	% Add bias to activation vector
    
end

y_h = W{2} * Xk;

%% FILL OUTPUT STRUCTURE

OUT.y_h = y_h;

%% END