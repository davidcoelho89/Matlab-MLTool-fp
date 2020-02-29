function [b0] = svc_f_bias(X,Yi,alpha,PAR)

% --- SVC bias calculate ---
%
%   [b0] = svc_f_bias(X,Yi,alpha,PAR)
%
%   Input:
%       X = input of support vectors                            [p x N]
%       Yi = output of support vectors                          [1 x N]
%       alpha = langrage multipliers                            [1 x N]
%       PAR.
%           epsilon = minimum value to be considered SV         [cte]
%           Ktype = kernel type ( see kernel_func() )           [cte]
%           sigma = kernel hyperparameter ( see kernel_func() )	[cte]
%           order = kernel hyperparameter ( see kernel_func() ) [cte]
%           alpha = kernel hyperparameter ( see kernel_func() ) [cte]
%           theta = kernel hyperparameter ( see kernel_func() ) [cte]
%           gamma = kernel hyperparameter ( see kernel_func() ) [cte]
%   Output:
%           b0 = optimum bias            	[cte]

%% INITIALIZATIONS

% Get Parameters
epsilon = PAR.epsilon;  % Used to considered a sample as a support vector

% Initialize Problem
[~,N] = size(X);        % number of samples

% Initialize Output
b0 = 0;                 % Optimum Bias

%% ALGORITHM

% Get one support vector for each class

svi = find(alpha > epsilon);
nsv = length(svi);

X_sv = X(:,svi);
Y_sv = Yi(svi);

for j = 1:nsv,
    if Y_sv(j) == 1,
        x_p = X_sv(:,j);
    else
        x_m = X_sv(:,j);
    end
end

% Calculate Optimum Bias

for j = 1:N,
    xj = X(:,j);
    b0 = b0 + alpha(j) * Yi(j) * ...
         (kernel_func(xj,x_p,PAR) + kernel_func(xj,x_m,PAR));
end

b0 = -0.5 * b0;

%% END