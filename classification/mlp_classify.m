function [OUT] = mlp_classify(DATA,PAR)

% --- MLP Classifier Test ---
%
%   [OUT] = mlp_classify(DATA,PAR)
%
%   Input:
%       DATA.
%           input = attributes                      [p x N]
%       PAR.
%           W{1} = Hidden layer weight Matrix   	[Nh x p+1]
%           W{2} = Output layer weight Matrix       [No x Nh+1]
%           Nh = number of hidden neurons         	[cte]
%           Nlin = Non-linearity                    [cte]
%               1 -> Sigmoid                        [0 e 1]
%               2 -> Hyperbolic Tangent             [-1 e +1]
%   Output:
%       OUT.
%           y_h = classifier's output               [No x N]

%% INITIALIZATIONS

% Get data
X = DATA.input;             % Get attributes matrix

% Get parameters
W{1} = PAR.W{1};          	% Hidden layer weight Matrix
W{2} = PAR.W{2};          	% Output layer weight Matrix
Nlin = PAR.Nlin;            % Non-linearity

% Problem Initialization
[No,~] = size(W{2});      	% Number of outputs
[~,N] = size(X);            % Number of samples

% Initialize Outputs
y_h = zeros(No,N);          % Estimated output

% Add bias to input matrix
X = [ones(1,N);X];          % x0 = +1

%% ALGORITHM

for t = 1:N

    % HIDDEN LAYER
    xi = X(:,t);              	% Get input sample
    Ui = W{1} * xi;            	% Activation of hidden neurons 
    Yi = mlp_f_ativ(Ui,Nlin);   % Non-linear function
    
    % OUTPUT LAYER
    xk = [+1; Yi];             	% build input of output layer
    Uk = W{2} * xk;            	% Activation of output neurons
    Yk = mlp_f_ativ(Uk,Nlin);	% Non-linear function
    
    y_h(:,t) = Yk;              % Hold neural net output

end

%% FILL OUTPUT STRUCTURE

OUT.y_h = y_h;

%% END