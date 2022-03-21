function [OUT] = mlp_predict(DATA,PAR)

% --- MLP Regression Test ---
%
%   [OUT] = mlp_classify(DATA,PAR)
%
%   Input:
%       DATA.
%           input = regression variables	[lag_u + lag_y x Ns-lag_max]
%       PAR.
%           W = weight matrices                     [NL x 1]
%               W{1:end-1} = Hidden layer weight Matrix 
%               W{end} = Output layer weight Matrix
%           Nh = number of hidden neurons         	[NL-1 x 1]
%           Nlin = Non-linearity                    [cte]
%               1 -> Sigmoid                        [0 e 1]
%           add_bias = add or not bias for input 	[0 or 1]
%               2 -> Hyperbolic Tangent             [-1 e +1]
%           prediction_type = type of prediction    [cte]
%           lag_output = lag for each output      	[1 x Ny]
%   Output:
%       OUT.
%           y_h = classifier's output               [No x N]

%% INITIALIZATIONS

% Get data
X = DATA.input;                     % Get attributes matrix

% Get parameters
W = PAR.W;                          % Weight Matrices
NL = length(W);                     % Number of layers
Nlin = PAR.Nlin;                    % Non-linearity
add_bias = PAR.add_bias;            % Add or not bias for input

pred_type = PAR.prediction_type;    % Type of prediction
lag_output = PAR.lag_output;        % Lag for each output

% Problem Initialization
[No,~] = size(W{NL});      	% Number of outputs
[~,N] = size(X);            % Number of samples

% Initialize Outputs
y_h = zeros(No,N);          % Estimated output

% Add bias to input matrix
if(add_bias == 1)
    X = [ones(1,N) ; X]; 	% x0 = +1
end

% Initialize memory of last predictions (for free simulation)
output_memory_length = sum(lag_output);
if(add_bias)
    output_memory = X(2:output_memory_length+1,1);
else
    output_memory = X(1:output_memory_length,1);
end

%% ALGORITHM

for n = 1:N
    
    xi = X(:,n);                      % Get input sample

    xi = update_regression_vector(xi,output_memory,pred_type,add_bias);
    
    for i = 1:NL
        Ui = W{i} * xi;               % Activation of hidden neurons
        if (i == NL)
            Yi = mlp_f_ativ(Ui,0);    % Layer Output (linear function)
        else
            Yi = mlp_f_ativ(Ui,Nlin); % Layer Output (Non-linear function)
        end
        xi = [+1; Yi];                % Build input for next layer
    end
    y_h(:,n) = Yi;                    % Get output of last layer
    
    output_memory = update_output_memory(Yi,output_memory,lag_output);

end

%% FILL OUTPUT STRUCTURE

OUT.y_h = y_h;

%% END