function [OUT] = krr_classify(DATA,PAR)

% --- Kernel Ridge Classifier Classify Function ---
%
%   [OUT] = krr_classify(DATA,PAR)
% 
%   Input:
%       DATA.
%           input = input matrix                                [p x N]
%       PAR.
%       	alphas = constants for the representer theorem      [N x Nc]
%           Xk = samples for the representer theorem            [p x N]
%           Ktype = kernel type ( see kernel_func() )           [cte]
%           sigma = kernel hyperparameter ( see kernel_func() ) [cte]
%           order = kernel hyperparameter ( see kernel_func() ) [cte]
%           alpha = kernel hyperparameter ( see kernel_func() ) [cte]
%           theta = kernel hyperparameter ( see kernel_func() ) [cte]
%           gamma = kernel hyperparameter ( see kernel_func() ) [cte]
%   Output:
%       OUT.
%           y_h = classifier's output             	[Nc x N]

%% INITIALIZATION

% Data Initialization
X = DATA.input;                 % Input matrix
[~,N] = size(X);                % Number of samples

% Get Hyperparameters

alphas = PAR.alphas;            % Constants for the representer theorem 
Xk = PAR.Xk;                    % samples for the representer theorem

% Problem Initilization
[Nc,~] = size(alphas);          % Number of prototypes and classes
[~,Nk] = size(Xk);           	% Number of kernel samples

% Init outputs
y_h = -1*ones(Nc,N);            % One output for each sample

%% ALGORITHM

for i = 1:N,
    % get sample
    x = X(:,i);
    % Calculate Kt
    kx = zeros(Nk,1);
    for j = 1:Nk,
        xj = Xk(:,j);
        kx(j) = kernel_func(x,xj,PAR);
    end
    % Calculate Outputs
   	for c = 1:Nc,
        alpha = alphas{c};
        y_h(c,i) = alpha'*kx;
    end    
end

%% FILL OUTPUT STRUCTURE

OUT.y_h = y_h;

%% END