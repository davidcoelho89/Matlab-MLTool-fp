function [K] = svc_f_kernel(X,Yi,PAR)

% --- SVC Kernel Function ---
%
% [K] = svc_f_kernel(X,Yi,PAR)
%
%   Input:
%       X = attributes                                          [p x N]
%       Yi = class                                              [1 x N]
%       PAR.
%           Ktype = kernel type ( see kernel_func() )           [cte]
%           sigma = kernel hyperparameter ( see kernel_func() ) [cte]
%           order = kernel hyperparameter ( see kernel_func() ) [cte]
%           alpha = kernel hyperparameter ( see kernel_func() ) [cte]
%           theta = kernel hyperparameter ( see kernel_func() ) [cte]
%           gamma = kernel hyperparameter ( see kernel_func() ) [cte]
%   Output:
%      K = Kernel Matrix                                        [N x N]

%% INITIALIZATION

% Initialize Problem
[~,N] = size(X);        % Number of samples

% Initialize Output
K = zeros(N,N);         % Kernel matrix

%% ALGORITHM

% Calculate Kernel Matrix
for i = 1:N
    for j = i:N
        K(i,j) = Yi(i) * Yi(j) * kernel_func(X(:,i),X(:,j),PAR);
        K(j,i) = K(i,j);
    end
end

% Avoid conditioning problems
lambda = (1e-10);
K = K + lambda*eye(N);

%% END