function [OUT] = linear_predict(DATA,PAR)

% --- Linear Regression Prediction ---
%
%   [OUT] = linear_predict(DATA,PAR)
%
%   Input:
%       DATA.
%           input = inputs matrix                       [p x N]
%       PAR.
%           W = transformation matrix                [No x p+1] or [No x p]
%           add_Bias = whether or not to add the bias   [0 or 1]
%   Output:
%       OUT.
%           y_h = estimated outputs matrix              [No x N]

%% INITIALIZATIONS

% Get data
X = DATA.input;             % input matrix

% Get parameters
W = PAR.W;                          % Weight matrix
add_bias = PAR.add_bias;            % whether or not to add the bias
pred_type = PAR.prediction_type;	% Type of prediction
lag_output = PAR.lag_output;        % Lag for each output

% Problem Initialization
[~,N] = size(X);                    % Number of samples and classes
[No,~] = size(W);                   % Number of outputs

% Initialize Outputs
y_h = zeros(No,N);                  % Estimated output

% Add bias to input matrix
if(add_bias)
    X = [ones(1,N) ; X];            % [x0 = +1]
end

% Initialize memory of last predictions (for free simulation)
output_memory_length = sum(lag_output);
if(add_bias)
    output_memory = X(2:output_memory_length+1,1);
else
    output_memory = X(1:output_memory_length,1);
end

%% ALGORITHM

% Function output
for n = 1:N
	xn = X(:,n);        % Get input sample
        
	xn = update_regression_vector(xn,output_memory,pred_type,add_bias);
        
	y_h(:,n) = W * xn;  % Generate Estimation
        
	output_memory = update_output_memory(y_h(:,n),output_memory,lag_output);
end

%% FILL OUTPUT STRUCTURE

OUT.y_h = y_h;

%% END