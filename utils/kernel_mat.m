function [Km] = kernel_mat(X,PAR)

% --- Calculate the Kernel Matrix of a data matrix ---
%
%   [Km] = kernel_mat(X,PAR)
%
%   Input:
%       X = matrix with samples                                 [p x N]
%       PAR.
%           Ktype = kernel type                                 [cte]
%               1 -> Linear
%               2 -> Gaussian (default)
%               3 -> Polynomial
%               4 -> Exponencial / Laplacian
%               5 -> Cauchy
%               6 -> Log
%               7 -> Sigmoid
%               8 -> Kmod
%           sigma   (gauss / exp / cauchy / log / kmod)         [cte]
%           order   (poly / log)                                [cte]
%           alpha   (poly / sigmoid)                            [cte]
%           theta   (lin / poly / sigmoid)                      [cte]
%           gamma   (Kmod)                                      [cte]
%   Output:
%       Km = kernel matrix, where Kij = K(X(:,i),X((:,j))       [N x N]

%% INITIALIZATIONS

% Get number of samples
[~,N] = size(X);

% Used to avoid inverse problems
if (~(isfield(PAR,'sig2n'))),
    PAR.sig2n = 0.001;
else
    sig2n = PAR.sig2n;
end

% Initialize Kernel Matrix
Km = zeros(N,N);

%% ALGORITHM

for i = 1:N,
    for j = i:N,
        Km(i,j) = kernel_func(X(:,i),X(:,j),PAR);
      	Km(j,i) = Km(i,j);
    end
end

Km = Km + sig2n*eye(N,N);

%% END