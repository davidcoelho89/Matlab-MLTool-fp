function [OUT] = mlp_predict(DATA,PAR)

% --- MLP Regression Test ---
%
%   [OUT] = mlp_classify(DATA,PAR)
%
%   Input:
%       DATA.
%           input = attributes                      [p x N]
%       PAR.
%           W = weight matrices                     [NL x 1]
%               W{1:end-1} = Hidden layer weight Matrix 
%               W{end} = Output layer weight Matrix
%           Nh = number of hidden neurons         	[NL-1 x 1]
%           Nlin = Non-linearity                    [cte]
%               1 -> Sigmoid                        [0 e 1]
%               2 -> Hyperbolic Tangent             [-1 e +1]
%           prediction_type = type of prediction    [0 or 1]
%           output_mem = memory buffer              []
%   Output:
%       OUT.
%           y_h = classifier's output               [No x N]

%% INITIALIZATIONS

% Get data
X = DATA.input';           	% Get attributes matrix

% Get parameters
W = PAR.W;                  % Weight Matrices
NL = length(W);             % Number of layers
Nlin = PAR.Nlin;            % Non-linearity

% Problem Initialization
[No,~] = size(W{NL});      	% Number of outputs
[~,N] = size(X);            % Number of samples

% Initialize Outputs
y_h = zeros(No,N);          % Estimated output

% Add bias to input matrix
X = [ones(1,N);X];          % x0 = +1

%% ALGORITHM

for t = 1:N
    
    xi = X(:,t);                      % Get input sample
    for i = 1:NL
        Ui = W{i} * xi;               % Activation of hidden neurons
        if (i == NL)
            Yi = mlp_f_ativ(Ui,0);    % Layer Output (linear function)
        else
            Yi = mlp_f_ativ(Ui,Nlin); % Layer Output (Non-linear function)
        end
        xi = [+1; Yi];                % Build input for next layer
    end
    y_h(:,t) = Yi;                    % Get output of last layer

end

%% FILL OUTPUT STRUCTURE

OUT.y_h = y_h';

%% END